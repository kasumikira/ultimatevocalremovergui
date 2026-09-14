"""Shared workflow policy for all separation backends."""

import os
import time
from dataclasses import dataclass, field, replace
from functools import wraps

import numpy as np

from gui_data.constants import (
    DONE,
    secondary_stem,
    INFERENCE_STEP_1,
    INFERENCE_STEP_1_PRE,
    INFERENCE_STEP_1_SEC,
    INFERENCE_STEP_1_VOC_S,
    INFERENCE_STEP_2_PRE,
    INFERENCE_STEP_2_PRE_CACHED_MODOEL,
    INFERENCE_STEP_2_PRIMARY_CACHED,
    INFERENCE_STEP_2_SEC,
    INFERENCE_STEP_2_SEC_CACHED_MODOEL,
    INFERENCE_STEP_2_VOC_S,
)
from lib_v5 import spec_utils
from separation.output import AudioOutput
from lib_v5.verify_gpu_availability import (
    GPU_TYPE_NVIDIA_CUDA, check_gpu_availability, onnxruntime_cuda_available,
)
from separation.request import RunRequest, RunServices, RunOptions, ALL_STEM_ROLES, StemSelection
from separation.state import (
    RunKind, RunContext, EnsembleContext, StemState, PitchPolicy, SecondaryPlan, DeviceState,
    ProgressState, VocalChain, OutputPolicy, BackendSettings,
    align_stem_selection, classify_run,
)


def cached_run(namespace):
    """Reuse this file's earlier run of the same model.

    Wraps a family's private inference step. The cache stores one opaque payload
    per ``(namespace, model)`` for the duration of one audio file, so the family
    hands over its output without the shared layer knowing what is inside it,
    and the family reads it back without writing any cache code of its own.
    ``namespace`` is the cache partition the family belongs to — the same key
    the application side uses, so MDX-Net and MDXC share one (see
    ``RunServices.load_cached_run``).
    """

    def decorate(infer):
        @wraps(infer)
        def infer_once(session, *args, **kwargs):
            cached = session.load_cached_run(namespace)
            if cached is not None:
                return cached
            payload = infer(session, *args, **kwargs)
            session.store_cached_run(namespace, payload)
            return payload

        return infer_once

    return decorate


@dataclass
class SeparationSession:
    """Declared run data with inference reporting and cache access.

    Model dispatch, secondary blending and vocal chains belong to the runner;
    audio transformations and saving belong to their explicit pipeline steps.
    """

    request: RunRequest
    services: RunServices
    context: RunContext
    pitch: PitchPolicy
    stems: StemState
    secondary_plan: SecondaryPlan | None
    device_state: DeviceState
    vocal_chain: VocalChain
    output_policy: OutputPolicy
    process_method: str
    model_path: str
    model_basename: str
    wav_type_set: str
    mp3_bit_set: str
    save_format: str
    DEVERBER_MODEL: str
    deverb_vocal_opt: str
    progress: ProgressState = field(default_factory=ProgressState)
    overwrite_protect_stamp: int = field(default_factory=lambda: round(time.time()))

    @property
    def audio_file(self):
        return self.request.audio_file

    @property
    def audio_file_base(self):
        return self.request.audio_file_base

    @property
    def export_path(self):
        return self.request.export_path

    def audio_file_base_voc_split(self, stem, split):
        base = self.audio_file_base.replace("_(Vocals)", "")
        return os.path.join(self.export_path, f'{base}_({stem}_{split}).wav')

    def stem_path(self, stem_name):
        return AudioOutput.stem_path(self, stem_name)

    def report_inference_done(self):
        """Close the current inference progress line."""
        self.services.write_to_console(DONE, base_text='')

    def load_cached_run(self, namespace):
        """This file's earlier run of this model, reported to the user."""
        if self.context.kind is RunKind.VOCAL_SPLIT:
            return None
        cached = self.services.load_cached_run(namespace, self.model_basename)
        if cached is None:
            return None
        self.report_cached_run()
        return cached

    def store_cached_run(self, namespace, payload):
        """Keep this run's output for the next run of the same model.

        Worth it only when this file's job invokes the model more than once, and
        never for a vocal-chain run: that one works on the vocal stem, not on
        this file's mixture, so it neither reads nor fills this cache.
        """
        if self.context.kind is RunKind.VOCAL_SPLIT:
            return
        if self.services.model_occurrences(self.model_basename) <= 1:
            return
        self.services.store_cached_run(namespace, payload, self.model_basename)

    def report_inference_start(self):
        if self.context.kind is RunKind.SECONDARY:
            self.services.write_to_console(INFERENCE_STEP_2_SEC(self.process_method, self.model_basename))

        if self.context.kind is RunKind.PREPROCESS:
            self.services.write_to_console(INFERENCE_STEP_2_PRE(self.process_method, self.model_basename))

        if self.context.kind is RunKind.VOCAL_SPLIT:
            self.services.write_to_console(INFERENCE_STEP_2_VOC_S(self.process_method, self.model_basename))

    def report_inference_running(self, is_no_write=False):
        self.services.write_to_console(DONE, base_text='') if not is_no_write else None
        self.services.set_progress_bar(0.05) if not is_no_write else None

        if self.context.kind is RunKind.SECONDARY:
            self.services.write_to_console(INFERENCE_STEP_1_SEC)
        elif self.context.kind is RunKind.PREPROCESS:
            self.services.write_to_console(INFERENCE_STEP_1_PRE)
        elif self.context.kind is RunKind.VOCAL_SPLIT:
            self.services.write_to_console(INFERENCE_STEP_1_VOC_S)
        else:
            self.services.write_to_console(INFERENCE_STEP_1)

    def report_inference_progress(self, length, is_match_mix=False):
        if not is_match_mix:
            self.progress.completed_steps += 1

            # Avoid division by zero
            if length <= 0:
                length = 1

            if (0.8/length*self.progress.completed_steps) >= 0.8:
                length = self.progress.completed_steps + 1

            self.services.set_progress_bar(
                0.1, (0.8/length*self.progress.completed_steps))

    def report_cached_run(self):
        """Tell the user this model is not being run again for this file."""
        if self.context.kind is RunKind.SECONDARY:
            self.services.write_to_console(INFERENCE_STEP_2_SEC_CACHED_MODOEL(self.process_method, self.model_basename))
        elif self.context.kind is RunKind.PREPROCESS:
            self.services.write_to_console(INFERENCE_STEP_2_PRE_CACHED_MODOEL(self.process_method, self.model_basename))
        else:
            self.services.write_to_console(INFERENCE_STEP_2_PRIMARY_CACHED, "")

    def pitch_fix(self, source, sr_pitched, org_mix):
        semitone_shift = self.pitch.semitones
        source = spec_utils.change_pitch_semitones(source, sr_pitched, semitone_shift=semitone_shift)[0]
        source = spec_utils.match_array_shapes(source, org_mix)
        return source

    def match_frequency_pitch(self, mix):
        source = mix
        if self.pitch.match_frequency and self.pitch.enabled:
            source, sr_pitched = spec_utils.change_pitch_semitones(mix, 44100, semitone_shift=-self.pitch.semitones)
            source = self.pitch_fix(source, sr_pitched, mix)

        return source


def create_run(request: RunRequest, services: RunServices,
                  family_setup, options: RunOptions | None = None):
    """Translate GUI configuration once and construct a fully initialized session.

    A family returns complete typed settings and its private runtime. No
    attributes are injected into an existing object and no string paths are
    interpreted during construction.
    """
    model = request.model
    options = options or RunOptions()
    kind = options.kind or classify_run(model)
    options = replace(options, kind=kind)
    device, gpu_type = check_gpu_availability(model.is_gpu_conversion, model.device_set)
    providers = ('CUDAExecutionProvider',) if (
        gpu_type == GPU_TYPE_NVIDIA_CUDA and onnxruntime_cuda_available
    ) else ('CPUExecutionProvider',)
    stems = StemState(
        primary=model.primary_stem,
        secondary=model.secondary_stem,
        ensemble_primary=model.ensemble_primary_stem,
        legacy_requested_roles=request.requested_stem_roles(
            child_run=kind is not RunKind.PRIMARY),
        selection=request.stem_selection,
        target_instrument=model.is_target_instrument,
    )
    if model.is_inst_only_voc_splitter or model.is_sec_bv_rebalance:
        stems = replace(stems, legacy_requested_roles=ALL_STEM_ROLES,
                        selection=StemSelection.all())
    if options.parent_stem and model.is_multi_stem_ensemble:
        stems = replace(stems, primary=options.parent_stem,
                        secondary=secondary_stem(options.parent_stem))
    settings = BackendSettings(
        stems=stems,
        device=DeviceState(model.device_set, device, gpu_type, providers),
        ensemble=EnsembleContext(model.is_ensemble_mode,
                                 request.is_4_stem_ensemble,
                                 model.is_multi_stem_ensemble),
    )
    settings, runtime = family_setup(model, request, settings, options)
    settings = align_stem_selection(settings, request, kind)
    session = SeparationSession(
        request=request,
        services=services,
        context=RunContext(kind, settings.ensemble),
        pitch=PitchPolicy(model.is_pitch_change, model.semitone_shift,
                          model.is_match_frequency_pitch),
        stems=settings.stems,
        secondary_plan=SecondaryPlan(model.secondary_model, model.secondary_model_scale)
            if kind is RunKind.PRIMARY and model.is_secondary_model_activated else None,
        device_state=settings.device,
        vocal_chain=VocalChain(model.vocal_split_model,
                               options.master_instrumental, options.master_vocal),
        output_policy=OutputPolicy(
            normalize=model.is_normalization,
            deverb_vocals=model.is_deverb_vocals,
            save_split_instrumental=(isinstance(options.master_instrumental, np.ndarray)
                                     and model.is_save_inst_vocal_splitter),
            split_instrumental_only=model.is_inst_only_voc_splitter,
            save_vocal_only=model.is_save_vocal_only,
            secondary_bv_rebalance=model.is_sec_bv_rebalance,
            karaoke=model.is_karaoke,
            backing_vocal_model=model.is_bv_model,
            backing_vocal_rebalanced=bool(model.bv_model_rebalance
                                         and kind is RunKind.VOCAL_SPLIT),
            save_preprocessed_instrumental=model.is_demucs_pre_proc_model_inst_mix,
        ),
        process_method=model.process_method,
        model_path=model.model_path,
        model_basename=model.model_basename,
        wav_type_set=model.wav_type_set,
        mp3_bit_set=model.mp3_bit_set,
        save_format=model.save_format,
        DEVERBER_MODEL=model.DEVERBER_MODEL,
        deverb_vocal_opt=model.deverb_vocal_opt,
    )
    return session, runtime
