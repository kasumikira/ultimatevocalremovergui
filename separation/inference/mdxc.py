from __future__ import annotations

import os
from dataclasses import dataclass, replace

import numpy as np
import soundfile as sf
import torch
from ml_collections import ConfigDict
from torch import nn

from gui_data.constants import (
    ALL_STEMS,
    DEFAULT,
    DEMUD_COMBINE_METHODS,
    DEMUD_PHASE_INVERT,
    DEMUD_PHASE_ROTATE,
    INST_STEM,
    MDX_ARCH_TYPE,
    VOCAL_STEM,
    secondary_stem,
)
from lib_v5 import spec_utils
from lib_v5.bandit.core.model import MultiMaskMultiSourceBandSplitRNNSimple
from lib_v5.bandit_v2.bandit import Bandit
from lib_v5.roformer.bs_roformer import BSRoformer
from lib_v5.roformer.bs_roformer_new import BSRoformer as BSRoformerNew
from lib_v5.roformer.mel_band_roformer import MelBandRoformer
from lib_v5.roformer.mel_band_roformer_new import MelBandRoformer as MelBandRoformerNew
from lib_v5.scnet.scnet import SCNet
from lib_v5.tfc_tdf_v3 import TFC_TDF_net
from lib_v5.verify_gpu_availability import GPU_TYPE_APPLE_MPS

from ..audio import prepare_mix
from ..contract import ProducedStems, SeparationBackend
from ..request import ALL_STEM_ROLES, RunKind, RunOptions
from ..state import BackendSettings, StemAlignment
from ..workflow import cached_run, create_run
from .vr import vr_denoiser

MDXC_MODELS = {
    'BS-Roformer': ('BSRoformer', 'model'),
    'BS-Roformer v2': ('BSRoformerNew', 'model'),
    'BS-Roformer New': ('BSRoformerNew', 'model'),
    'MelBand-Roformer': ('MelBandRoformer', 'model'),
    'MelBand-Roformer v2': ('MelBandRoformerNew', 'model'),
    'MelBand-Roformer New': ('MelBandRoformerNew', 'model'),
    'SCNet': ('SCNet', 'model'),
    'Bandit': ('MultiMaskMultiSourceBandSplitRNNSimple', 'model'),
    'Bandit v2': ('Bandit', 'kwargs'),
    'Bandit 2': ('Bandit', 'kwargs'),
}


def build_mdxc_model(model_type, config, device, is_roformer=False, constructors=None):
    """Construct a network without loading weights or running inference.

    Unknown legacy types retain the original MDX23C/config-based fallback.
    The optional namespace lets callers retain their existing constructor bindings.
    """
    if constructors is None:
        constructors = globals()
    entry = MDXC_MODELS.get(model_type)
    if entry is not None:
        symbol, section = entry
        return constructors[symbol](**getattr(config, section))
    if is_roformer:
        if 'num_bands' in config.model:
            return constructors['MelBandRoformer'](**config.model)
        if 'freqs_per_bands' in config.model:
            return constructors['BSRoformer'](**config.model)
        raise ValueError('Unknown model type in the configuration.')
    return constructors['TFC_TDF_net'](config, device=device)


class MDXCInference:
    def _windowing_array(window_size: int, fade_size: int, device) -> torch.Tensor:
        fadein = torch.linspace(0, 1, fade_size).to(device)
        fadeout = torch.linspace(1, 0, fade_size).to(device)
        window = torch.ones(window_size).to(device)
        window[-fade_size:] = fadeout
        window[:fade_size] = fadein
        return window


    def overlap_add(session, result, x, l, j, start, window):
        if session.device_state.gpu_type == GPU_TYPE_APPLE_MPS:
            x = x.to(session.device_state.torch_device)
        result[..., start:start + l] += x[j, ..., :l] * window[..., :l]
        return result


    def find_hop_size(self, config_):
        def _find_hop(config):
            if isinstance(config, dict) or isinstance(config, ConfigDict):
                for key, value in config.items():
                    if key in ('hop_size', 'hop_length'):
                        return value
                    elif isinstance(value, (dict, ConfigDict)):
                        result = _find_hop(value)
                        if result is not None:
                            return result
            elif hasattr(config, '__dict__'):
                return _find_hop(config.__dict__)
        set_chunk = config_.audio.chunk_size
        dim_t_c = getattr(config_.inference, 'dim_t', 256) - 1
        hop_size = _find_hop(config_)
        if hop_size and hop_size * dim_t_c == set_chunk:
            return hop_size
        else:
            return set_chunk // dim_t_c


    def _load_model(session, runtime):
        """Keep checkpoint interpretation and strict loading compatible."""
        model = build_mdxc_model(
            runtime.mdx_model_type, runtime.config, session.device_state.torch_device,
            runtime.is_roformer,
        )
        checkpoint = torch.load(session.model_path, map_location='cpu')
        model = model if not isinstance(model, torch.nn.DataParallel) else model.module
        model.load_state_dict(checkpoint)
        model.to(session.device_state.torch_device).eval()
        return model


    def _predict_chunks(session, runtime, model, mix, chunk_size, num_overlap, num_instruments, chunk_add):
        """Run the legacy overlap-add algorithm; no loading or stem processing."""
        device = session.device_state.torch_device
        step = int(chunk_size // num_overlap)
        fade_size = chunk_size // 10
        border = chunk_size - step
        batch_size = 1
        length_init = mix.shape[-1]
        windowing_array = MDXCInference._windowing_array(
            chunk_size, fade_size, device)
        if length_init > 2 * border and border:
            mix = nn.functional.pad(mix, (border, border), mode='reflect')
        batch_len = int(mix.shape[1] / step)
        if runtime.is_demud:
            batch_len = batch_len * chunk_add
        with torch.inference_mode() if runtime.is_use_torch_inference_mode else torch.no_grad():
            req_shape = (num_instruments,) + mix.shape
            result = torch.zeros(req_shape, dtype=torch.float32, device=device)
            counter = torch.zeros(req_shape, dtype=torch.float32, device=device)
            batch_data = []
            batch_locations = []
            i = 0
            while i < mix.shape[1]:
                part = mix[:, i:i + chunk_size].to(device)
                length = part.shape[-1]
                if length > chunk_size // 2:
                    pad_mode = 'reflect'
                else:
                    pad_mode = 'constant'
                part = nn.functional.pad(part, (0, chunk_size - length), mode=pad_mode, value=0)
                batch_data.append(part)
                batch_locations.append((i, length))
                i += step
                # Process in batches
                if len(batch_data) >= batch_size or (i >= mix.shape[1]):
                    arr = torch.stack(batch_data, dim=0)
                    x = model(arr)
                    window = windowing_array.clone()
                    if i - step == 0:
                        window[:fade_size] = 1
                    elif i >= mix.shape[1]:
                        window[-fade_size:] = 1
                    for j, (start, seg_len) in enumerate(batch_locations):
                        session.report_inference_progress(batch_len)
                        min_length = min(result[..., start:start + seg_len].shape[-1], x[j, ..., :seg_len].shape[-1], window[..., :seg_len].shape[-1])
                        result = MDXCInference.overlap_add(
                            session, result, x, min_length, j, start, window)
                        counter[..., start:start + seg_len] += window[..., :seg_len]
                    batch_data = []
                    batch_locations = []
            # Normalize by the overlap counter and remove padding
            estimated_sources = result / counter.clamp(min=1e-10)
            if length_init > 2 * (chunk_size - step) and chunk_size - step > 0:
                estimated_sources = estimated_sources[..., chunk_size - step:-(chunk_size - step)]
            estimated_sources = estimated_sources.cpu().numpy()
        return estimated_sources


@dataclass
class MDXCRuntime:
    """Config-driven MDX23C/Roformer-family inference and stem derivation."""

    config: ConfigDict
    mdx_model_type: str
    is_roformer: bool
    mdx_segment_size: int | str
    overlap_mdx23: int
    is_force_mdx_c_seg_def: bool
    is_use_torch_inference_mode: bool
    mdxnet_stem_select: str
    is_mdx_combine_stems: bool
    is_vocal_main_target: bool
    is_denoise_model: bool
    denoiser_model: str | None
    is_demud: bool
    demudder_method: str
    invert_spectrogram: bool
    primary_audio: np.ndarray | None = None
    secondary_audio: np.ndarray | None = None


def mdxc_runtime(model_data) -> MDXCRuntime:
    """Read MDXC configuration without requiring MDX-Net or ONNX parameters."""
    config = model_data.mdx_c_configs
    return MDXCRuntime(
        config=config,
        mdx_model_type=model_data.mdx_model_type,
        is_roformer=model_data.is_roformer,
        mdx_segment_size=model_data.mdx_segment_size,
        overlap_mdx23=model_data.overlap_mdx23,
        is_force_mdx_c_seg_def=model_data.is_force_mdx_c_seg_def,
        is_use_torch_inference_mode=model_data.is_use_torch_inference_mode,
        mdxnet_stem_select=model_data.mdxnet_stem_select,
        is_mdx_combine_stems=model_data.is_mdx_combine_stems,
        is_vocal_main_target=config.training.target_instrument == VOCAL_STEM,
        is_denoise_model=model_data.is_denoise_model,
        denoiser_model=model_data.DENOISER_MODEL,
        is_demud=model_data.is_demud,
        demudder_method=model_data.demudder_method,
        invert_spectrogram=model_data.is_invert_spec,
    )


def mdxc_settings(model_data, request, settings: BackendSettings) -> BackendSettings:
    """Resolve MDXC's ensemble pair independently of MDX-Net settings."""
    stems = replace(settings.stems, alignment=StemAlignment.TARGET_ONLY)
    if not request.is_4_stem_ensemble and not model_data.is_target_instrument:
        stems = replace(
            stems,
            primary=(model_data.ensemble_primary_stem
                     if request.is_ensemble_master else model_data.primary_stem),
            secondary=(model_data.ensemble_secondary_stem
                       if request.is_ensemble_master else model_data.secondary_stem),
        )
    return replace(settings, stems=stems)


def family_setup(model_data, request, settings: BackendSettings,
                 options: RunOptions):
    """Construct MDXC state and resolve any requested secondary-model pair."""
    settings = mdxc_settings(model_data, request, settings)
    runtime = mdxc_runtime(model_data)
    if options.kind in (RunKind.SECONDARY, RunKind.PREPROCESS):
        stem_list = native_stem_list(runtime)
        select = (stem_list[0] if options.kind is RunKind.PREPROCESS
                  else options.target_stem or options.parent_stem
                  or model_data.primary_model_primary_stem)
        runtime.mdxnet_stem_select = select
        complement = secondary_stem(select)
        # When the request names a derived complement (for example
        # Instrumental) and its native counterpart exists (Vocals), orient the
        # immutable pair around the native prediction. The requested stem is
        # then derived as the secondary without renaming state during inference.
        primary, secondary = (
            (complement, select) if complement in stem_list
            else (select, complement)
        )
        settings = replace(settings, stems=replace(
            settings.stems, primary=primary, secondary=secondary,
            legacy_requested_roles=ALL_STEM_ROLES))
    return settings, runtime

@cached_run(MDX_ARCH_TYPE)
def _infer(session, runtime):
    """Run this model for this file and return the mixture and its predictions.

    Private to the family, and what the cache holds: this model's output for
    this audio.
    """
    session.report_inference_start()
    session.report_inference_running()
    mix = prepare_mix(session.audio_file)
    sources = demix(session, runtime, mix)
    session.report_inference_done()

    return mix, sources

def native_stem_list(runtime) -> list:
    """The stem names this model can produce."""
    config = runtime.config.training
    if config.target_instrument and config.target_instrument != VOCAL_STEM:
        return [config.target_instrument]
    return list(config.instruments)

def uses_stem_list(session, runtime) -> bool:
    """Whether this run writes the model's whole named stem list."""
    stem_list = native_stem_list(runtime)
    is_ensemble_4_stem = (session.context.ensemble.multi_stem_output
                          and not len(stem_list) <= 2)
    return bool(
        (runtime.mdxnet_stem_select == ALL_STEMS
         and not session.request.is_ensemble_master
         and not len(stem_list) <= 2
         and session.context.kind is RunKind.PRIMARY)
        or (is_ensemble_4_stem and session.context.kind is not RunKind.PREPROCESS))

def available_stems(session, runtime) -> tuple:
    if uses_stem_list(session, runtime):
        return tuple(native_stem_list(runtime))
    return (session.stems.primary, session.stems.secondary)

def produce_stems(session, runtime):
    mix, sources = _infer(session, runtime)
    wanted = session.stems.select(available_stems(session, runtime))
    if uses_stem_list(session, runtime):
        return ProducedStems(stems={stem: sources[stem].T for stem in wanted})
    return produce_dual_stems(session, runtime, mix, sources, wanted)

def produce_dual_stems(session, runtime, mix, sources, wanted):
    source_primary = select_primary_output(session, runtime, sources)
    stems = {}
    # Derive the complement before exposing the model's native prediction.
    if session.stems.secondary in wanted:
        stems[session.stems.secondary] = secondary_stem_audio(
            session, runtime, mix, sources, source_primary)
    if session.stems.primary in wanted:
        stems[session.stems.primary] = primary_stem_audio(
            session, runtime, source_primary)
    return ProducedStems(stems=stems)

def select_primary_output(session, runtime, sources):
    stem_list = native_stem_list(runtime)
    if len(stem_list) == 1:
        return sources
    if session.context.ensemble.per_stem_routing or len(stem_list) == 2:
        if session.context.kind is not RunKind.PRIMARY and session.stems.primary in stem_list:
            return sources[session.stems.primary]
        return sources[stem_list[0]]
    if session.context.kind is RunKind.VOCAL_SPLIT:
        # The GUI labels these outputs lead_only/backing_only; those labels
        # are not keys in the model's native prediction dictionary.
        return sources[runtime.mdxnet_stem_select]
    if session.context.kind is not RunKind.PRIMARY and len(stem_list) > 2:
        return sources[session.stems.primary]
    return sources[runtime.mdxnet_stem_select]

def primary_stem_audio(session, runtime, source_primary):
    if not isinstance(runtime.primary_audio, np.ndarray):
        runtime.primary_audio = source_primary.T
    return runtime.primary_audio

def secondary_stem_audio(session, runtime, mix, sources, source_primary):
    if isinstance(runtime.secondary_audio, np.ndarray):
        return runtime.secondary_audio

    stem_list = native_stem_list(runtime)
    if (isinstance(sources, dict) and len(stem_list) == 2
            and session.stems.secondary not in sources):
        is_mdx_combine_stems = False
    else:
        is_mdx_combine_stems = runtime.is_mdx_combine_stems
    if is_mdx_combine_stems and len(stem_list) >= 2:
        if len(stem_list) == 2:
            secondary_source = sources[session.stems.secondary]
        else:
            # Derived rather than popped: this model's output may be the one a
            # later run of the same model reads back from the run cache.
            native_primary = (runtime.mdxnet_stem_select
                              if session.context.kind is RunKind.VOCAL_SPLIT
                              else session.stems.primary)
            combined = [value for name, value in sources.items()
                        if name != native_primary]
            secondary_source = np.zeros_like(combined[0])
            for value in combined:
                secondary_source += value
        runtime.secondary_audio = secondary_source.T
    else:
        runtime.secondary_audio, raw_mix = source_primary, session.match_frequency_pitch(mix)
        runtime.secondary_audio = spec_utils.to_shape(runtime.secondary_audio, raw_mix.shape)
        if runtime.invert_spectrogram:
            runtime.secondary_audio = spec_utils.invert_stem(raw_mix, runtime.secondary_audio)
        else:
            runtime.secondary_audio = (-runtime.secondary_audio.T+raw_mix.T)
    return runtime.secondary_audio

def demix(session, runtime, mix, is_demud=False):
    sr_pitched = 441000
    org_mix = mix
    chunk_add = 3 if runtime.demudder_method == DEMUD_COMBINE_METHODS else 2
    if session.pitch.enabled:
        mix, sr_pitched = spec_utils.change_pitch_semitones(mix, 44100, semitone_shift=-session.pitch.semitones)
    device = session.device_state.torch_device
    model = MDXCInference._load_model(session, runtime)
    mix = torch.tensor(mix, dtype=torch.float32, device=session.device_state.torch_device)
    hop_size = MDXCInference.find_hop_size(session, runtime.config)
    is_seg_def = True if runtime.mdx_segment_size == DEFAULT else False
    if is_seg_def or runtime.is_force_mdx_c_seg_def:
        chunk_size = runtime.config.audio.chunk_size
    else:
        chunk_size = hop_size * (runtime.mdx_segment_size - 1)
    is_target_inst = False
    num_instruments = 1 if runtime.config.training.target_instrument else len(runtime.config.training.instruments)
    num_overlap = runtime.overlap_mdx23
    target_stem = [runtime.config.training.target_instrument] if runtime.config.training.target_instrument else None
    if num_instruments == 1:
        is_target_inst = True if runtime.config.training.target_instrument == INST_STEM else False
    estimated_sources = MDXCInference._predict_chunks(
        session, runtime,
        model, mix, chunk_size, num_overlap, num_instruments, chunk_add,
    )
    pitch_fix = lambda s:session.pitch_fix(s, sr_pitched, org_mix)
    if num_instruments > 1 or runtime.is_vocal_main_target or is_target_inst:
        sources = {k: pitch_fix(v) if session.pitch.enabled else v for k, v in zip(target_stem if target_stem else runtime.config.training.instruments, estimated_sources)}
        if runtime.is_vocal_main_target:
            if sources[VOCAL_STEM].shape[1] != org_mix.shape[1]:
                sources[VOCAL_STEM] = spec_utils.match_array_shapes(sources[VOCAL_STEM], org_mix)
            sources[INST_STEM] = org_mix - sources[VOCAL_STEM]
        if is_target_inst:
            if sources[INST_STEM].shape[1] != org_mix.shape[1]:
                sources[INST_STEM] = spec_utils.match_array_shapes(sources[INST_STEM], org_mix)
            sources[VOCAL_STEM] = org_mix - sources[INST_STEM]
        if runtime.is_denoise_model and VOCAL_STEM in sources.keys() and INST_STEM in sources.keys():
            sources[VOCAL_STEM] = vr_denoiser(sources[VOCAL_STEM], session.device_state.torch_device, model_path=runtime.denoiser_model)
            if sources[VOCAL_STEM].shape[1] != org_mix.shape[1]:
                sources[VOCAL_STEM] = spec_utils.match_array_shapes(sources[VOCAL_STEM], org_mix)
            sources[INST_STEM] = org_mix - sources[VOCAL_STEM]
        if is_demud:
            return sources[VOCAL_STEM]
        elif VOCAL_STEM in sources.keys() and INST_STEM in sources.keys() and runtime.is_demud:
            session.services.write_to_console(
                'De-mudding Instrumental stem... ', base_text='')
            inst_source = spec_utils.match_array_shapes(sources[INST_STEM], org_mix)
            if runtime.demudder_method == DEMUD_COMBINE_METHODS:
                bare_source = org_mix - inst_source
                phase_app_mix = spec_utils.demud_processor(org_mix, inst_source, demudder_method=DEMUD_PHASE_ROTATE)
                phase_app_inv_mix = spec_utils.demud_processor(org_mix, inst_source, demudder_method=DEMUD_PHASE_INVERT)
                phase_stem_remix = demix(
                    session, runtime, phase_app_mix, is_demud=True)
                phase_stem_inv = demix(
                    session, runtime, phase_app_inv_mix, is_demud=True)
                bare_stem_list = [bare_source, phase_stem_remix, phase_stem_inv]
                phase_stem = spec_utils.average_audio(bare_stem_list, is_demud=True)
            else:
                phase_app_mix = spec_utils.demud_processor(org_mix, inst_source, demudder_method=runtime.demudder_method)
                phase_stem = demix(
                    session, runtime, phase_app_mix, is_demud=True)
            try:
                sf.write(os.path.join('demud_tests', f'{session.audio_file_base}_(phased_mix).wav'), phase_app_mix.T, 44100, subtype=session.wav_type_set)
            except:
                print('Failed to save inverted file')
            sources[INST_STEM] = org_mix - spec_utils.match_array_shapes(phase_stem, org_mix)
        if is_target_inst and VOCAL_STEM in sources.keys():
            sources = sources[INST_STEM]
        return sources
    else:
        sources = {k: v for k, v in zip([runtime.config.training.target_instrument], estimated_sources)}
        est_s = sources[runtime.config.training.target_instrument]
        return pitch_fix(est_s) if session.pitch.enabled else est_s


def create_backend(request, services, options=None):
    """This family wired for one run: its state, plus the functions the
    shared runner calls.

    Nothing is loaded or computed here: the family's model runs when the
    runner calls ``produce_stems``.
    """
    session, runtime = create_run(request, services, family_setup, options)
    return SeparationBackend(
        session=session,
        runtime=runtime,
        produce_stems=produce_stems,
    )
