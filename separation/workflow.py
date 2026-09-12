"""Shared workflow policy for all separation backends."""

import numpy as np

from gui_data.constants import (
    DEMUCS_ARCH_TYPE, DONE, INFERENCE_STEP_1, INFERENCE_STEP_1_PRE,
    INFERENCE_STEP_1_SEC, INFERENCE_STEP_1_VOC_S, INFERENCE_STEP_2_PRE,
    INFERENCE_STEP_2_PRE_CACHED_MODOEL, INFERENCE_STEP_2_PRIMARY_CACHED,
    INFERENCE_STEP_2_SEC, INFERENCE_STEP_2_SEC_CACHED_MODOEL,
    INFERENCE_STEP_2_VOC_S, INST_STEM, MDX_ARCH_TYPE, VOCAL_STEM,
    VR_ARCH_TYPE,
)
from lib_v5 import spec_utils
from separation.orchestration import process_chain_model
from separation.output import AudioOutput
from separation.state import SeparationState


class SeperateAttributes(SeparationState, AudioOutput):

    def check_label_secondary_stem_runs(self):
        if (self.process_data['is_ensemble_master'] and not self.is_4_stem_ensemble and not self.is_mdx_c) or (self.process_data['is_ensemble_master'] and self.is_target_instrument):
            if self.ensemble_primary_stem != self.primary_stem:
                self.is_primary_stem_only, self.is_secondary_stem_only = self.is_secondary_stem_only, self.is_primary_stem_only

        if self.is_pre_proc_model or self.is_secondary_model:
            self.is_primary_stem_only = False
            self.is_secondary_stem_only = False

    def start_inference_console_write(self):
        if self.is_secondary_model and not self.is_pre_proc_model and not self.is_vocal_split_model:
            self.write_to_console(INFERENCE_STEP_2_SEC(self.process_method, self.model_basename))

        if self.is_pre_proc_model:
            self.write_to_console(INFERENCE_STEP_2_PRE(self.process_method, self.model_basename))

        if self.is_vocal_split_model:
            self.write_to_console(INFERENCE_STEP_2_VOC_S(self.process_method, self.model_basename))

    def running_inference_console_write(self, is_no_write=False):
        self.write_to_console(DONE, base_text='') if not is_no_write else None
        self.set_progress_bar(0.05) if not is_no_write else None

        if self.is_secondary_model and not self.is_pre_proc_model and not self.is_vocal_split_model:
            self.write_to_console(INFERENCE_STEP_1_SEC)
        elif self.is_pre_proc_model:
            self.write_to_console(INFERENCE_STEP_1_PRE)
        elif self.is_vocal_split_model:
            self.write_to_console(INFERENCE_STEP_1_VOC_S)
        else:
            self.write_to_console(INFERENCE_STEP_1)

    def running_inference_progress_bar(self, length, is_match_mix=False):
        if not is_match_mix:
            self.progress_value += 1

            # Avoid division by zero
            if length <= 0:
                length = 1

            if (0.8/length*self.progress_value) >= 0.8:
                length = self.progress_value + 1

            self.set_progress_bar(0.1, (0.8/length*self.progress_value))

    def load_cached_sources(self):
        if self.is_secondary_model and not self.is_pre_proc_model:
            self.write_to_console(INFERENCE_STEP_2_SEC_CACHED_MODOEL(self.process_method, self.model_basename))
        elif self.is_pre_proc_model:
            self.write_to_console(INFERENCE_STEP_2_PRE_CACHED_MODOEL(self.process_method, self.model_basename))
        else:
            self.write_to_console(INFERENCE_STEP_2_PRIMARY_CACHED, "")

    def cache_source(self, secondary_sources):
        model_occurrences = self.list_all_models.count(self.model_basename)

        if not model_occurrences <= 1:
            if self.process_method == MDX_ARCH_TYPE:
                self.cached_model_source_holder(MDX_ARCH_TYPE, secondary_sources, self.model_basename)

            if self.process_method == VR_ARCH_TYPE:
                self.cached_model_source_holder(VR_ARCH_TYPE, secondary_sources, self.model_basename)

            if self.process_method == DEMUCS_ARCH_TYPE:
                self.cached_model_source_holder(DEMUCS_ARCH_TYPE, secondary_sources, self.model_basename)

    def process_vocal_split_chain(self, sources: dict):
        def is_valid_vocal_split_condition(master_vocal_source):
            """Checks if conditions for vocal split processing are met."""
            conditions = [
                isinstance(master_vocal_source, np.ndarray),
                self.vocal_split_model,
                not self.is_ensemble_mode,
                not self.is_karaoke,
                not self.is_bv_model
            ]
            return all(conditions)

        master_inst_source = sources.get(INST_STEM, None)
        master_vocal_source = sources.get(VOCAL_STEM, None)

        if is_valid_vocal_split_condition(master_vocal_source):
            process_chain_model(
                self.vocal_split_model,
                self.process_data,
                vocal_stem_path=self.master_vocal_path,
                master_vocal_source=master_vocal_source,
                master_inst_source=master_inst_source
            )

    def process_secondary_stem(self, stem_source, secondary_model_source=None, model_scale=None):
        if not self.is_secondary_model:
            if self.is_secondary_model_activated and isinstance(secondary_model_source, np.ndarray):
                secondary_model_scale = model_scale if model_scale else self.secondary_model_scale
                stem_source = spec_utils.average_dual_sources(stem_source, secondary_model_source, secondary_model_scale)

        return stem_source

    def final_process(self, stem_path, source, secondary_source, stem_name, samplerate):
        source = self.process_secondary_stem(source, secondary_source)
        self.write_audio(stem_path, source, samplerate, stem_name=stem_name)

        return {stem_name: source}

    def pitch_fix(self, source, sr_pitched, org_mix):
        semitone_shift = self.semitone_shift
        source = spec_utils.change_pitch_semitones(source, sr_pitched, semitone_shift=semitone_shift)[0]
        source = spec_utils.match_array_shapes(source, org_mix)
        return source

    def match_frequency_pitch(self, mix):
        source = mix
        if self.is_match_frequency_pitch and self.is_pitch_change:
            source, sr_pitched = spec_utils.change_pitch_semitones(mix, 44100, semitone_shift=-self.semitone_shift)
            source = self.pitch_fix(source, sr_pitched, mix)

        return source
