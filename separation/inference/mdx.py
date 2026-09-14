from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import onnxruntime as ort
import torch
from onnx import load
from onnx2pytorch import ConvertModel

import lib_v5.mdxnet as MdxnetSet
from gui_data.constants import (
    DEFAULT,
    DEMUD_COMBINE_METHODS,
    DEMUD_PHASE_INVERT,
    DEMUD_PHASE_ROTATE,
    INST_STEM,
    MDX_ARCH_TYPE,
    MDX_NET_FREQ_CUT,
    NO_STEM,
    PRIMARY_STEM,
    SECONDARY_STEM,
)
from lib_v5 import spec_utils
from lib_v5.tfc_tdf_v3 import STFT
from lib_v5.verify_gpu_availability import onnxruntime_cuda_available

from ..audio import prepare_mix
from ..contract import ProducedStems, SeparationBackend
from ..request import RunKind, RunOptions
from ..state import BackendSettings, StemAlignment
from ..workflow import cached_run, create_run
from .vr import vr_denoiser


class MDXInference:
    def prepare_stft(session, runtime):
        runtime.trim = runtime.n_fft//2
        runtime.chunk_size = runtime.hop * (runtime.mdx_segment_size-1)
        runtime.stft = STFT(runtime.n_fft, runtime.hop, runtime.dim_f, session.device_state.torch_device)


    def demix(session, runtime, mix, is_match_mix=False, is_demud=False):
        MDXInference.prepare_stft(session, runtime)
        
        is_bare_stem = NO_STEM not in runtime.native_primary and not runtime.native_primary == INST_STEM
        org_mix = mix
        tar_waves_ = []

        comp_valu = runtime.compensate
        is_calculate_comp = False if is_match_mix or is_bare_stem else runtime.is_calculate_comp
        chunk_add = 3 if runtime.demudder_method == DEMUD_COMBINE_METHODS else 2
        if is_match_mix:
            chunk_size = runtime.hop * (256-1)
            overlap = 0.02
        else:
            chunk_size = runtime.chunk_size
            overlap = runtime.overlap_mdx
            
            if session.pitch.enabled:
                mix, sr_pitched = spec_utils.change_pitch_semitones(mix, 44100, semitone_shift=-session.pitch.semitones)

        gen_size = chunk_size-2*runtime.trim

        pad = gen_size + runtime.trim - ((mix.shape[-1]) % gen_size)
        mixture = np.concatenate((np.zeros((2, runtime.trim), dtype='float32'), mix, np.zeros((2, pad), dtype='float32')), 1)

        step = runtime.chunk_size - runtime.n_fft if overlap == DEFAULT else int((1 - overlap) * chunk_size)
        result = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
        divider = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
        total = 0
        total_chunks = (mixture.shape[-1] + step - 1) // step

        if runtime.is_demud:
            total_chunks = total_chunks * chunk_add
            print('Total Chunks: ', total_chunks)

        for i in range(0, mixture.shape[-1], step):
            total += 1
            start = i
            end = min(i + chunk_size, mixture.shape[-1])

            chunk_size_actual = end - start

            if overlap == 0:
                window = None
            else:
                window = np.hanning(chunk_size_actual)
                window = np.tile(window[None, None, :], (1, 2, 1))

            mix_part_ = mixture[:, start:end]
            if end != i + chunk_size:
                pad_size = (i + chunk_size) - end
                mix_part_ = np.concatenate((mix_part_, np.zeros((2, pad_size), dtype='float32')), axis=-1)

            mix_part = torch.tensor([mix_part_], dtype=torch.float32).to(session.device_state.torch_device)
            mix_waves = mix_part.split(1)
            
            with torch.no_grad():
                for mix_wave in mix_waves:
                    session.report_inference_progress(total_chunks, is_match_mix=is_match_mix)

                    tar_waves = MDXInference.predict_chunk(
                        session, runtime, mix_wave, is_match_mix=is_match_mix)
                    
                    if window is not None:
                        tar_waves[..., :chunk_size_actual] *= window 
                        divider[..., start:end] += window
                    else:
                        divider[..., start:end] += 1

                    result[..., start:end] += tar_waves[..., :end-start]
            
        tar_waves = result / divider
        tar_waves_.append(tar_waves)

        tar_waves_ = np.vstack(tar_waves_)[:, :, runtime.trim:-runtime.trim]
        tar_waves = np.concatenate(tar_waves_, axis=-1)[:, :mix.shape[-1]]
        
        source = tar_waves[:,0:None]

        if session.pitch.enabled and not is_match_mix:
            source = session.pitch_fix(source, sr_pitched, org_mix)

        if is_calculate_comp:
            comp_valu = spec_utils.calculate_comp_level(spec_utils.match_array_shapes(source, org_mix), org_mix, comp_set=runtime.compensate)
        source = source if is_match_mix else source*comp_valu

        if runtime.is_denoise_model and not is_match_mix:
            if NO_STEM in runtime.native_primary or runtime.native_primary == INST_STEM:
                if org_mix.shape[1] != source.shape[1]:
                    source = spec_utils.match_array_shapes(source, org_mix)
                source = org_mix - vr_denoiser(org_mix-source, session.device_state.torch_device, model_path=runtime.denoiser_model)
            else:
                source = vr_denoiser(source, session.device_state.torch_device, model_path=runtime.denoiser_model)

        if is_demud:
            source = spec_utils.match_array_shapes(source, org_mix)
            source = source if is_bare_stem else org_mix - source
            return source

        if (runtime.is_demud and not is_match_mix
                and session.context.kind is not RunKind.VOCAL_SPLIT):
            session.services.write_to_console(
                "De-mudding Instrumental stem... ", base_text="")
            inst_source = org_mix - spec_utils.match_array_shapes(source, org_mix) if is_bare_stem else source
            inst_source = spec_utils.match_array_shapes(inst_source, org_mix)
            bare_source = org_mix - inst_source
            if runtime.demudder_method == DEMUD_COMBINE_METHODS:
                phase_app_mix = spec_utils.demud_processor(
                    org_mix, inst_source, demudder_method=DEMUD_PHASE_ROTATE
                )
                phase_app_inv_mix = spec_utils.demud_processor(
                    org_mix, inst_source, demudder_method=DEMUD_PHASE_INVERT
                )
                phase_stem_remix = MDXInference.demix(
                    session, runtime, phase_app_mix, is_demud=True)
                phase_stem_inv = MDXInference.demix(
                    session, runtime, phase_app_inv_mix, is_demud=True)
                bare_stem_list = [bare_source, phase_stem_remix, phase_stem_inv]
                phase_stem = spec_utils.average_audio(
                    bare_stem_list, is_demud=True
                )
            else:
                phase_app_mix = spec_utils.demud_processor(
                    org_mix, inst_source, demudder_method=runtime.demudder_method
                )
                phase_stem = MDXInference.demix(
                    session, runtime, phase_app_mix, is_demud=True)
            source = phase_stem if is_bare_stem else org_mix - spec_utils.match_array_shapes(phase_stem, org_mix)
            if is_bare_stem:
                source = {PRIMARY_STEM: bare_source, SECONDARY_STEM: source}
            else:
                source = {PRIMARY_STEM: source, SECONDARY_STEM: bare_source}
        return source


    def predict_chunk(session, runtime, mix, is_match_mix=False):
        
        spek = runtime.stft(mix.to(session.device_state.torch_device))*runtime.adjust
        spek[:, :, :3, :] *= 0 

        if is_match_mix:
            spec_pred = spek.cpu().numpy()
        else:
            spec_pred = -runtime.model_run(-spek)*0.5+runtime.model_run(spek)*0.5 if runtime.is_denoise else runtime.model_run(spek)

        return runtime.stft.inverse(torch.tensor(spec_pred).to(session.device_state.torch_device)).cpu().detach().numpy()


@dataclass
class MDXRuntime:
    """MDX-Net checkpoint/ONNX parameters and STFT inference state."""

    is_mdx_ckpt: bool
    is_denoise: bool
    is_denoise_model: bool
    denoiser_model: str | None
    compensate: float
    mdx_segment_size: int | str
    is_calculate_comp: bool
    is_demud: bool
    demudder_method: str
    overlap_mdx: float | str
    n_fft: int
    dim_f: int
    dim_t: int
    native_primary: str
    invert_spectrogram: bool
    adjust: int = 1
    hop: int = 1024
    primary_audio: np.ndarray | None = None
    secondary_audio: np.ndarray | None = None
    model_run: object = None
    stft: STFT | None = None
    trim: int = 0
    chunk_size: int = 0


def mdx_runtime(model_data) -> MDXRuntime:
    """Read only the GUI fields that configure MDX-Net inference."""
    return MDXRuntime(
        is_mdx_ckpt=model_data.is_mdx_ckpt,
        is_denoise=model_data.is_denoise,
        is_denoise_model=model_data.is_denoise_model,
        denoiser_model=model_data.DENOISER_MODEL,
        compensate=model_data.compensate,
        mdx_segment_size=model_data.mdx_segment_size,
        is_calculate_comp=model_data.is_calculate_comp,
        is_demud=model_data.is_demud,
        demudder_method=model_data.demudder_method,
        overlap_mdx=model_data.overlap_mdx,
        n_fft=model_data.mdx_n_fft_scale_set,
        dim_f=model_data.mdx_dim_f_set,
        dim_t=2 ** model_data.mdx_dim_t_set,
        native_primary=model_data.primary_stem_native,
        invert_spectrogram=model_data.is_invert_spec,
    )


def family_setup(model_data, request, settings: BackendSettings,
                 options: RunOptions):
    """Align MDX-Net's pair and construct its own runtime."""
    settings = replace(settings, stems=replace(
        settings.stems, alignment=StemAlignment.ENSEMBLE_PAIR))
    return settings, mdx_runtime(model_data)


@cached_run(MDX_ARCH_TYPE)
def _infer(session, runtime):
    """Run this model for this file and return the mixture and its prediction.

    Private to the family, and what the cache holds: this model's output for
    this audio. Anything a single run derives on top of it stays outside.
    """
    if runtime.mdx_segment_size == DEFAULT:
        runtime.mdx_segment_size = runtime.dim_t

    session.report_inference_start()

    if runtime.is_mdx_ckpt:
        model_params = torch.load(session.model_path, map_location=lambda storage, loc: storage)['hyper_parameters']
        runtime.hop = model_params['hop_length']
        separator = MdxnetSet.ConvTDFNet(**model_params)
        runtime.model_run = separator.load_from_checkpoint(session.model_path).to(session.device_state.torch_device).eval()
    else:
        if runtime.mdx_segment_size == runtime.dim_t and onnxruntime_cuda_available:
            ort_ = ort.InferenceSession(session.model_path, providers=session.device_state.onnx_providers)
            runtime.model_run = lambda spek:ort_.run(None, {'input': spek.cpu().numpy()})[0]
        else:
            runtime.model_run = ConvertModel(load(session.model_path))
            runtime.model_run.to(session.device_state.torch_device).eval()

    session.report_inference_running()
    mix = prepare_mix(session.audio_file)

    source = MDXInference.demix(session, runtime, mix)

    session.report_inference_done()
    return mix, source

def produce_stems(session, runtime):
    """Produce the stems the caller asked for, in the caller's order."""
    try:
        wanted = session.stems.select(
            (session.stems.primary, session.stems.secondary))
        mix, source = _infer(session, runtime)
        # Derived per run, not cached: which stems this invocation targets
        # decides whether the complement needs a frequency-matched second pass.
        is_frequency_cut = session.stems.primary in MDX_NET_FREQ_CUT and session.pitch.match_frequency
        if not isinstance(source, np.ndarray) and type(source) is dict:
            runtime.secondary_audio = MDXInference.demix(
                session, runtime, source[SECONDARY_STEM], is_match_mix=True).T if is_frequency_cut else source[SECONDARY_STEM].T
            source = source[PRIMARY_STEM]
        stems = {}
        for stem_name in wanted:
            if stem_name == session.stems.primary:
                stems[stem_name] = primary_stem_audio(session, runtime, source)
            elif stem_name == session.stems.secondary:
                stems[stem_name] = secondary_stem_audio(
                    session, runtime, mix, source, is_frequency_cut)
        return ProducedStems(stems=stems)
    finally:
        runtime.model_run = None

def primary_stem_audio(session, runtime, source):
    if not isinstance(runtime.primary_audio, np.ndarray):
        runtime.primary_audio = source.T
    return runtime.primary_audio

def secondary_stem_audio(session, runtime, mix, source, is_frequency_cut):
    if not isinstance(runtime.secondary_audio, np.ndarray):
        raw_mix = MDXInference.demix(
            session, runtime, session.match_frequency_pitch(mix),
            is_match_mix=True) if is_frequency_cut else session.match_frequency_pitch(mix)
        runtime.secondary_audio = spec_utils.invert_stem(raw_mix, source) if runtime.invert_spectrogram else raw_mix.T-source.T
    return runtime.secondary_audio


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
