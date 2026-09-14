"""Explicit model execution, secondary blending, postprocessing and output."""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np

from gui_data.constants import INST_STEM, VOCAL_STEM
from lib_v5 import spec_utils
from lib_v5.verify_gpu_availability import clear_gpu_cache
from separation.contract import ProducedStems, SeparationBackend, SeparationResult, StemChild
from separation.output import AudioOutput
from separation.postprocess import prepare_output
from separation.request import RunKind, RunOptions


class SeparationRunner:
    def __init__(self, backend: SeparationBackend):
        self.backend = backend

    @property
    def session(self):
        return self.backend.session

    @property
    def runtime(self):
        return self.backend.runtime

    def run(self) -> SeparationResult:
        # Backends own the lifetime of their loaded model. Always clear unused
        # accelerator allocations on failure as well as on normal completion.
        try:
            produced = self.backend.produce_stems(self.session, self.runtime)
            self.validate_stems(produced)
            self.apply_secondary_models(produced)
            self.write_produced_stems(produced)
            self.run_vocal_chain(produced.stems)
            return SeparationResult(produced.stems)
        finally:
            clear_gpu_cache()

    def validate_stems(self, produced: ProducedStems):
        selection = self.session.stems.selection
        if selection is not None and selection.names is not None:
            if set(produced.stems) != selection.names:
                raise ValueError(f'Backend returned {tuple(produced.stems)}, requested {selection.names}')
        if produced.stems.keys() & produced.extra_stems.keys():
            raise ValueError('Export-only artifacts must have distinct names')
        for name, audio in (produced.stems | produced.extra_stems).items():
            if not isinstance(audio, np.ndarray) or audio.ndim != 2 or audio.shape[1] != 2:
                raise ValueError(f'{name}: expected sample-major stereo audio at 44100 Hz')

    def secondary_run_for(self, stem_name: str) -> StemChild | None:
        if self.session.context.kind is not RunKind.PRIMARY:
            return None
        # A family with per-stem routing supplies the whole policy, including
        # its ordinary pair case; None means no child, never implicit fallback.
        if self.backend.secondary_run_for is not None:
            return self.backend.secondary_run_for(self.session, self.runtime, stem_name)
        plan = self.session.secondary_plan
        if plan is None or plan.model is None:
            return None
        return StemChild(
            plan.model,
            options=RunOptions(kind=RunKind.SECONDARY,
                               parent_method=self.session.process_method,
                               parent_stem=self.session.stems.primary),
            scale=plan.scale,
            use_stem=stem_name,
        )

    def apply_secondary_models(self, produced: ProducedStems):
        runs = {}
        for stem_name, source in produced.stems.items():
            child = self.secondary_run_for(stem_name)
            if child is None:
                continue
            key = (id(child.model), child.options)
            if key not in runs:
                runs[key] = run_secondary_model(
                    child.model, self.session.request, self.session.services, child.options)
            contribution = self.pick_child_result(runs[key], child)
            if contribution is not None:
                scale = child.scale
                if scale is None and self.session.secondary_plan is not None:
                    scale = self.session.secondary_plan.scale
                produced.stems[stem_name] = spec_utils.average_dual_sources(
                    source, contribution, scale)

    @staticmethod
    def pick_child_result(result, child: StemChild):
        # Exact names only: "No Vocals" must never be mistaken for "Vocals".
        if child.use_stem is None or not isinstance(result, dict):
            return None
        return result.get(child.use_stem)

    def write_produced_stems(self, produced: ProducedStems) -> dict:
        artifacts = prepare_output(self.session, produced.stems | produced.extra_stems)
        AudioOutput().write_artifacts(self.session, artifacts)
        return produced.stems

    def vocal_chain_input(self, stems: dict) -> dict | None:
        vocal = stems.get(VOCAL_STEM)
        if not isinstance(vocal, np.ndarray):
            return None
        instrumental = stems.get(INST_STEM)
        if isinstance(instrumental, np.ndarray):
            return {VOCAL_STEM: vocal, INST_STEM: instrumental}
        if self.session.output_policy.secondary_bv_rebalance:
            return None
        return {VOCAL_STEM: vocal}

    def run_vocal_chain(self, stems: dict):
        session = self.session
        # A child computes input for its parent. Only the completed top-level
        # result is eligible for another vocal processing pass.
        if (session.context.kind is not RunKind.PRIMARY
                or not session.vocal_chain.model
                or session.context.ensemble.shared_output
                or session.output_policy.karaoke
                or session.output_policy.backing_vocal_model):
            return
        sources = self.vocal_chain_input(stems)
        if sources is None:
            return
        vocal_path = session.stem_path(VOCAL_STEM)
        session.vocal_chain.master_vocal_path = vocal_path
        run_vocal_splitter(
            session.vocal_chain.model, session.request, session.services,
            vocal_path, sources[VOCAL_STEM], sources.get(INST_STEM))


def run_secondary_model(model, request, services, options: RunOptions | None = None):
    """Execute a child on the same input and return exactly named stems."""
    from separation.factory import run_separator

    services.process_iteration()
    child_request = replace(request, model=model, stem_selection=None)
    options = replace(options or RunOptions(), kind=RunKind.SECONDARY)
    return run_separator(child_request, services, options).stems


def run_vocal_splitter(model, request, services, vocal_stem_path,
                       master_vocal_source, master_inst_source=None):
    """Execute a child on the completed vocal, explicitly outside file writing."""
    from separation.factory import run_separator

    services.process_iteration()
    vocal_source = master_vocal_source
    if model.bv_model_rebalance:
        vocal_source = spec_utils.reduce_mix_bv(
            master_inst_source, master_vocal_source,
            reduction_rate=model.bv_model_rebalance)
    child_request = replace(
        request, model=model, audio_file=vocal_source,
        audio_file_base=os.path.splitext(os.path.basename(vocal_stem_path))[0],
        stem_selection=None)
    options = RunOptions(kind=RunKind.VOCAL_SPLIT,
                         master_instrumental=master_inst_source,
                         master_vocal=master_vocal_source)
    return run_separator(child_request, services, options).stems
