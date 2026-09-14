from __future__ import annotations

import gzip
import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from demucs.apply import apply_model, demucs_segments
from demucs.hdemucs import HDemucs
from demucs.model_v2 import auto_load_demucs_model_v2
from demucs.pretrained import get_model as _gm
from demucs.utils import apply_model_v1, apply_model_v2
from gui_data.constants import (
    ALL_STEMS,
    DEMUCS_2_SOURCE_MAPPER,
    DEMUCS_4_SOURCE_MAPPER,
    DEMUCS_6_SOURCE_MAPPER,
    DEMUCS_ARCH_TYPE,
    DEMUCS_V1,
    DEMUCS_V2,
    DEMUCS_V3,
    DEMUCS_V4,
    GUITAR_STEM,
    INST_STEM,
    MDX_ARCH_TYPE,
    OTHER_STEM,
    PIANO_STEM,
    VOCAL_STEM,
    VR_ARCH_TYPE,
    secondary_stem,
)
from lib_v5 import spec_utils
from lib_v5.verify_gpu_availability import GPU_TYPE_APPLE_MPS, clear_gpu_cache

from ..audio import prepare_mix
from ..contract import ProducedStems, SeparationBackend, StemChild
from ..request import RunKind, RunOptions
from ..state import BackendSettings
from ..workflow import cached_run, create_run

cpu = torch.device('cpu')


class DemucsInference:
    def demix(session, runtime, mix):
        
        org_mix = mix
        
        if session.pitch.enabled:
            mix, sr_pitched = spec_utils.change_pitch_semitones(mix, 44100, semitone_shift=-session.pitch.semitones)
        
        processed = {}
        mix = torch.tensor(mix, dtype=torch.float32)
        ref = mix.mean(0)        
        mix = (mix - ref.mean()) / ref.std()
        mix_infer = mix 
        
        with torch.no_grad():
            if runtime.demucs_version == DEMUCS_V1:
                sources = apply_model_v1(runtime.demucs, 
                                            mix_infer.to(session.device_state.torch_device),
                                            runtime.shifts, 
                                            runtime.is_split_mode,
                                            set_progress_bar=session.services.set_progress_bar)
            elif runtime.demucs_version == DEMUCS_V2:
                sources = apply_model_v2(runtime.demucs, 
                                            mix_infer.to(session.device_state.torch_device),
                                            runtime.shifts,
                                            runtime.is_split_mode,
                                            runtime.overlap,
                                            set_progress_bar=session.services.set_progress_bar)
            else:
                print('runtime.shifts', runtime.shifts)
                print('is_split_mode', runtime.is_split_mode)
                print('overlap', runtime.overlap)
                sources = apply_model(runtime.demucs, 
                                        mix_infer[None], 
                                        runtime.shifts,
                                        runtime.is_split_mode,
                                        runtime.overlap,
                                        static_shifts=1 if runtime.shifts == 0 else runtime.shifts,
                                        set_progress_bar=session.services.set_progress_bar,
                                        device=session.device_state.torch_device)[0]
        
        sources = (sources * ref.std() + ref.mean()).cpu().numpy()
        sources[[0,1]] = sources[[1,0]]
        processed[mix] = sources[:,:,0:None].copy()
        sources = list(processed.values())
        sources = [s[:,:,0:None] for s in sources]
        #sources = [session.pitch_fix(s[:,:,0:None], sr_pitched, org_mix) if session.pitch.enabled else s[:,:,0:None] for s in sources]
        sources = np.concatenate(sources, axis=-1)
                     
        if session.pitch.enabled:
            sources = np.stack([session.pitch_fix(stem, sr_pitched, org_mix) for stem in sources])
                        
        return sources


@dataclass
class DemucsRuntime:
    """What one Demucs run owns for itself.

    The model configuration it runs with, what this run was asked to answer,
    and the network it loads. None of it belongs to the shared run state.
    """

    #: how this model runs
    demucs_version: str
    segment: int
    demucs_source_list: list
    demucs_source_map: dict
    is_custom_demucs: bool
    is_demucs_combine_stems: bool
    pre_proc_model: object
    shifts: int
    overlap: float
    is_split_mode: bool
    demucs_stems: str | None
    #: The per-stem secondary models the application configures, indexed by
    #: ``demucs_source_map``; empty when this run has none.
    secondary_models_by_stem: tuple = ()
    secondary_scales_by_stem: tuple = ()
    #: whether this run answers with the model's named stem list rather than a
    #: primary/complement pair. The caller asks for it; this family overrules it
    #: for a multi-stem run that is not a sub-run.
    wants_stem_list: bool = False
    #: what one run accumulates
    invert_spectrogram: bool = False
    primary_audio: object = None
    secondary_audio: object = None
    #: what the inference loads
    demucs: object = None


def family_setup(model_data, request, settings: BackendSettings,
                 options: RunOptions):
    """The shared state this family aligns, and the state it owns itself."""
    demucs_version = model_data.demucs_version
    demucs_source_map = model_data.demucs_source_map
    is_secondary_model = options.kind in (RunKind.SECONDARY, RunKind.PREPROCESS)

    device = replace(
        settings.device,
        torch_device=(cpu if settings.device.gpu_type == GPU_TYPE_APPLE_MPS
                      and demucs_version not in [DEMUCS_V3, DEMUCS_V4]
                      else settings.device.torch_device),
    )
    stems = replace(
        settings.stems,
        primary=(model_data.ensemble_primary_stem
                 if request.is_ensemble_master else model_data.primary_stem),
        secondary=(model_data.ensemble_secondary_stem
                   if request.is_ensemble_master else model_data.secondary_stem),
    )
    runtime = DemucsRuntime(
        demucs_version=demucs_version,
        segment=model_data.segment,
        demucs_source_list=model_data.demucs_source_list,
        demucs_source_map=demucs_source_map,
        is_custom_demucs=model_data.is_custom_demucs,
        is_demucs_combine_stems=model_data.is_demucs_combine_stems,
        pre_proc_model=model_data.pre_proc_model,
        shifts=model_data.shifts,
        overlap=model_data.overlap,
        is_split_mode=True if demucs_version == DEMUCS_V4 else model_data.is_split_mode,
        demucs_stems=(None if options.parent_method in [MDX_ARCH_TYPE, VR_ARCH_TYPE]
                      else model_data.demucs_stems),
        secondary_models_by_stem=tuple(model_data.secondary_model_4_stem),
        secondary_scales_by_stem=tuple(model_data.secondary_model_4_stem_scale),
        wants_stem_list=options.wants_stem_list,
        invert_spectrogram=model_data.is_invert_spec,
    )

    if (model_data.is_multi_stem_ensemble or request.is_4_stem_ensemble) and not is_secondary_model:
        runtime.wants_stem_list = True

    if model_data.is_multi_stem_ensemble and options.parent_stem:
        settings = replace(settings, ensemble=replace(
            settings.ensemble, multi_stem_output=False))
        if options.parent_stem in demucs_source_map:
            stems = replace(stems, primary=options.parent_stem,
                            secondary=secondary_stem(options.parent_stem))
        elif secondary_stem(options.parent_stem) in demucs_source_map:
            stems = replace(stems, primary=secondary_stem(options.parent_stem),
                            secondary=options.parent_stem)

    if is_secondary_model and not request.is_ensemble_master:
        requested_primary = (options.target_stem or options.parent_stem
                             or model_data.primary_model_primary_stem)
        if model_data.demucs_stem_count != 2 and requested_primary == INST_STEM:
            stems = replace(stems, primary=VOCAL_STEM, secondary=INST_STEM)
        else:
            stems = replace(stems, primary=requested_primary,
                            secondary=secondary_stem(requested_primary))

    return replace(settings, stems=stems, device=device), runtime

@cached_run(DEMUCS_ARCH_TYPE)
def _infer(session, runtime):
    """Run this model for this file and return its output.

    Private to the family: the model's own inference result is not part of the
    interface the caller sees. The cached unit is this whole run, including the
    pre-processed instrumental when the model has a pre-processor, so a later
    run of the same model reads it back rather than running either model again.
    """
    inst_mix = None
    inst_source = None
    is_no_write = False
    is_no_piano_guitar = False

    session.report_inference_start()
    mix = prepare_mix(session.audio_file)

    try:
        if runtime.demucs_version == DEMUCS_V1:
            if str(session.model_path).endswith(".gz"):
                with gzip.open(session.model_path, "rb") as checkpoint:
                    klass, args, kwargs, state = torch.load(checkpoint)
            else:
                klass, args, kwargs, state = torch.load(session.model_path)
            runtime.demucs = klass(*args, **kwargs)
            runtime.demucs.to(session.device_state.torch_device)
            runtime.demucs.load_state_dict(state)
        elif runtime.demucs_version == DEMUCS_V2:
            runtime.demucs = auto_load_demucs_model_v2(runtime.demucs_source_list, session.model_path)
            runtime.demucs.to(session.device_state.torch_device)
            runtime.demucs.load_state_dict(torch.load(session.model_path))
            runtime.demucs.eval()
        else:
            print('demucs_source_list', runtime.demucs_source_list)
            print('model_path', session.model_path)
            print('segment', runtime.segment)
            runtime.demucs = HDemucs(sources=runtime.demucs_source_list)
            runtime.demucs = _gm(name=os.path.splitext(os.path.basename(session.model_path))[0],
                              repo=Path(os.path.dirname(session.model_path)))
            runtime.demucs = demucs_segments(runtime.segment, runtime.demucs)
            runtime.demucs.to(session.device_state.torch_device)
            runtime.demucs.eval()

        if runtime.pre_proc_model:
            if session.stems.primary not in [VOCAL_STEM, INST_STEM]:
                is_no_write = True
                session.report_inference_done()
                mix_no_voc = run_pre_proc_model(session, runtime)
                inst_mix = prepare_mix(mix_no_voc[INST_STEM])
                session.services.process_iteration()
                session.report_inference_running(is_no_write=is_no_write)
                inst_source = DemucsInference.demix(session, runtime, inst_mix)
                session.services.process_iteration()

        if not runtime.pre_proc_model:
            session.report_inference_running(is_no_write=is_no_write)
        source = DemucsInference.demix(session, runtime, mix)
        session.report_inference_done()
    finally:
        runtime.demucs = None
        clear_gpu_cache()

    if isinstance(inst_source, np.ndarray):
        source_reshape = spec_utils.reshape_sources(inst_source[runtime.demucs_source_map[VOCAL_STEM]], source[runtime.demucs_source_map[VOCAL_STEM]])
        inst_source[runtime.demucs_source_map[VOCAL_STEM]] = source_reshape
        source = inst_source

    if isinstance(source, np.ndarray):

        if not runtime.is_custom_demucs:
            # Actual checkpoint output determines standard model names.
            # Custom models keep their configured mapping.
            if len(source) == 2:
                runtime.demucs_source_map = DEMUCS_2_SOURCE_MAPPER
            else:
                runtime.demucs_source_map = DEMUCS_6_SOURCE_MAPPER if len(source) == 6 else DEMUCS_4_SOURCE_MAPPER

                if (len(source) == 6 and session.request.is_ensemble_master
                        or len(source) == 6 and session.context.kind is not RunKind.PRIMARY):
                    is_no_piano_guitar = True
                    six_stem_other_source = list(source)
                    six_stem_other_source = [i for n, i in enumerate(source) if n in [runtime.demucs_source_map[OTHER_STEM], runtime.demucs_source_map[GUITAR_STEM], runtime.demucs_source_map[PIANO_STEM]]]
                    other_source = np.zeros_like(six_stem_other_source[0])
                    for i in six_stem_other_source:
                        other_source += i
                    source_reshape = spec_utils.reshape_sources(source[runtime.demucs_source_map[OTHER_STEM]], other_source)
                    source[runtime.demucs_source_map[OTHER_STEM]] = source_reshape

    return mix, source, inst_mix, is_no_piano_guitar

def run_pre_proc_model(session, runtime):
    """Run the pre-processor and return its named stems.

    This is this task's own first step — its instrumental output is the audio
    the main model runs on — so the backend launches it directly rather than
    going through the caller's secondary-model policy.
    """
    from separation.factory import run_separator

    child_request = replace(
        session.request, model=runtime.pre_proc_model, stem_selection=None)
    return run_separator(
        child_request, session.services,
        RunOptions(kind=RunKind.PREPROCESS)).stems

def uses_stem_list(session, runtime) -> bool:
    """Whether this run writes the model's whole named stem list."""
    return bool((runtime.demucs_stems == ALL_STEMS and not session.request.is_ensemble_master)
                or (session.context.ensemble.multi_stem_output
                    and runtime.wants_stem_list))

def available_stems(session, runtime) -> tuple:
    if uses_stem_list(session, runtime):
        return tuple(runtime.demucs_source_map.keys())
    return (session.stems.primary, session.stems.secondary)

def secondary_run_for(session, runtime, stem_name):
    """A multi-stem run gives every stem its own secondary model."""
    if not uses_stem_list(session, runtime):
        plan = session.secondary_plan
        if not plan or not plan.model:
            return None
        return StemChild(
            plan.model,
            options=RunOptions(kind=RunKind.SECONDARY,
                               parent_stem=session.stems.primary,
                               parent_method=DEMUCS_ARCH_TYPE),
            scale=plan.scale,
            use_stem=stem_name,
        )
    index = runtime.demucs_source_map.get(stem_name)
    if index is None or index >= len(runtime.secondary_models_by_stem):
        return None
    model = runtime.secondary_models_by_stem[index]
    if not model:
        return None
    return StemChild(
        model,
        options=RunOptions(kind=RunKind.SECONDARY, target_stem=stem_name,
                           parent_method=DEMUCS_ARCH_TYPE,
                           wants_stem_list=True),
        scale=runtime.secondary_scales_by_stem[index],
        use_stem=child_stem_for(model, stem_name, index),
    )

def child_stem_for(secondary_model, stem_name, stem_index):
    """Which of a child model's own stems this step uses.

    A two-source child always contributes its vocal output; a larger one
    contributes the stem at the same index. Both used to be list indices.
    """
    if secondary_model.process_method != DEMUCS_ARCH_TYPE:
        return stem_name
    if secondary_model.demucs_stem_count == 2:
        return VOCAL_STEM
    return next((name for name, index in secondary_model.demucs_source_map.items()
                 if index == stem_index), stem_name)

def produce_stems(session, runtime):
    mix, source, inst_mix, is_no_piano_guitar = _infer(session, runtime)
    # Configuration can advertise four stems for a six-source checkpoint.
    # Derive names from the actual payload on cache hits as well as fresh runs.
    if not runtime.is_custom_demucs:
        runtime.demucs_source_map = (
            DEMUCS_2_SOURCE_MAPPER if len(source) == 2 else
            DEMUCS_6_SOURCE_MAPPER if len(source) == 6 else DEMUCS_4_SOURCE_MAPPER)
    wanted = session.stems.select(available_stems(session, runtime))
    if uses_stem_list(session, runtime):
        stems = {name: source[index].T
                 for name, index in runtime.demucs_source_map.items()
                 if name in wanted}
        # One name per stem of this model: written, and what the parent run
        # receives when it asks for a stem.
        return ProducedStems(stems=stems)
    return produce_dual_stems(
        session, runtime, mix, source, inst_mix, is_no_piano_guitar, wanted)

def produce_dual_stems(session, runtime, mix, source, inst_mix, is_no_piano_guitar, wanted):
    results = {}
    if session.stems.secondary in wanted:
        results[session.stems.secondary] = derive_secondary_stem(
            session, runtime, source, mix, is_no_piano_guitar)

    extra_stems = {}
    if (session.stems.secondary in wanted
            and session.output_policy.save_preprocessed_instrumental and runtime.pre_proc_model
            and not session.context.ensemble.multi_stem_output):
        extra_stems[f'{session.stems.secondary} {INST_STEM}'] = derive_inst_mix(
            session, runtime, source, inst_mix, is_no_piano_guitar)

    if session.stems.primary in wanted:
        results[session.stems.primary] = primary_stem_audio(
            session, runtime, source)
    return ProducedStems(stems=results, extra_stems=extra_stems)

def primary_stem_audio(session, runtime, source):
    if not isinstance(runtime.primary_audio, np.ndarray):
        runtime.primary_audio = source[runtime.demucs_source_map[session.stems.primary]].T
    return runtime.primary_audio

def derive_secondary_stem(session, runtime, source, raw_mixture, is_no_piano_guitar):
    if isinstance(runtime.secondary_audio, np.ndarray):
        return runtime.secondary_audio
    runtime.secondary_audio = derive_from_sources(
        session, runtime,
        source, raw_mixture, is_no_piano_guitar, is_inst_mixture=False)
    return runtime.secondary_audio

def derive_inst_mix(session, runtime, source, raw_mixture, is_no_piano_guitar):
    return derive_from_sources(
        session, runtime, source, raw_mixture, is_no_piano_guitar,
        is_inst_mixture=True)

def derive_from_sources(session, runtime, source, raw_mixture, is_no_piano_guitar, is_inst_mixture):
    """Build one stem out of the model's own sources."""
    if runtime.is_demucs_combine_stems:
        source = list(source)
        if is_inst_mixture:
            source = [i for n, i in enumerate(source)
                      if not n in [runtime.demucs_source_map[session.stems.primary],
                                   runtime.demucs_source_map[VOCAL_STEM]]]
        else:
            source.pop(runtime.demucs_source_map[session.stems.primary])

        source = source[:len(source) - 2] if is_no_piano_guitar else source
        combined = np.zeros_like(source[0])
        for i in source:
            combined += i
        return combined.T

    if not isinstance(raw_mixture, np.ndarray):
        raw_mixture = prepare_mix(session.audio_file)

    derived = source[runtime.demucs_source_map[session.stems.primary]]
    if runtime.invert_spectrogram:
        return spec_utils.invert_stem(raw_mixture, derived)
    raw_mixture = spec_utils.reshape_sources(derived, raw_mixture)
    return (-derived.T + raw_mixture.T)


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
        secondary_run_for=secondary_run_for,
    )
