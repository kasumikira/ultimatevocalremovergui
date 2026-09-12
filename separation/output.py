from __future__ import annotations

import os

import soundfile as sf

from .audio import save_format
from .inference.vr import vr_denoiser
from gui_data.constants import (
    BV_VOCAL_STEM,
    BV_VOCAL_STEM_I,
    BV_VOCAL_STEM_LABEL,
    DONE,
    INFERENCE_STEP_DEVERBING,
    INST_STEM,
    LEAD_VOCAL_STEM,
    LEAD_VOCAL_STEM_I,
    LEAD_VOCAL_STEM_LABEL,
    SAVING_STEM,
    VOCAL_STEM,
)
from lib_v5 import spec_utils


class AudioOutput:
    def write_audio(self, stem_path: str, stem_source, samplerate, stem_name=None):
        
        def save_audio_file(path: str, source):
            source = spec_utils.normalize(source, self.is_normalization)

            if os.path.isfile(path) and is_not_ensemble:
                path = path.replace(".wav", f"_{self.overwrite_protect_stamp}.wav")

            sf.write(path, source, samplerate, subtype=self.wav_type_set)

            if is_not_ensemble:
                save_format(path, self.save_format, self.mp3_bit_set)

        def save_voc_split_instrumental(stem_name, stem_source, is_inst_invert=False):
            inst_stem_name = "Instrumental (With Lead Vocals)" if stem_name == LEAD_VOCAL_STEM else "Instrumental (With Backing Vocals)"
            inst_stem_path_name = LEAD_VOCAL_STEM_I if stem_name == LEAD_VOCAL_STEM else BV_VOCAL_STEM_I
            inst_stem_path = self.audio_file_base_voc_split(INST_STEM, inst_stem_path_name)
            stem_source = -stem_source if is_inst_invert else stem_source
            inst_stem_source = spec_utils.combine_arrarys([self.master_inst_source, stem_source], is_swap=True)
            save_with_message(inst_stem_path, inst_stem_name, inst_stem_source)

        def save_voc_split_vocal(stem_name, stem_source):
            voc_split_stem_name = LEAD_VOCAL_STEM_LABEL if stem_name == LEAD_VOCAL_STEM else BV_VOCAL_STEM_LABEL
            voc_split_stem_path = self.audio_file_base_voc_split(VOCAL_STEM, stem_name)
            save_with_message(voc_split_stem_path, voc_split_stem_name, stem_source)

        def save_with_message(stem_path, stem_name, stem_source):
            is_deverb = self.is_deverb_vocals and (
                self.deverb_vocal_opt == stem_name or
                (self.deverb_vocal_opt == 'ALL' and 
                (stem_name == VOCAL_STEM or stem_name == LEAD_VOCAL_STEM_LABEL or stem_name == BV_VOCAL_STEM_LABEL)))

            self.write_to_console(f'{SAVING_STEM[0]}{stem_name}{SAVING_STEM[1]}')
            
            if is_deverb and is_not_ensemble:
                deverb_vocals(stem_path, stem_source)
            
            save_audio_file(stem_path, stem_source)
            self.write_to_console(DONE, base_text='')
            
        def deverb_vocals(stem_path:str, stem_source):
            self.write_to_console(INFERENCE_STEP_DEVERBING, base_text='')
            stem_source_deverbed, stem_source_2 = vr_denoiser(stem_source, self.device, is_deverber=True, model_path=self.DEVERBER_MODEL)
            save_audio_file(stem_path.replace(".wav", "_deverbed.wav"), stem_source_deverbed)
            save_audio_file(stem_path.replace(".wav", "_reverb_only.wav"), stem_source_2)
            
        is_bv_model_lead = (self.is_bv_model_rebalenced and self.is_vocal_split_model and stem_name == LEAD_VOCAL_STEM)
        is_bv_rebalance_lead = (self.is_bv_model_rebalenced and self.is_vocal_split_model and stem_name == BV_VOCAL_STEM)
        is_no_vocal_save = self.is_inst_only_voc_splitter and (stem_name == VOCAL_STEM or stem_name == BV_VOCAL_STEM or stem_name == LEAD_VOCAL_STEM) or is_bv_model_lead
        is_not_ensemble = (not self.is_ensemble_mode or self.is_vocal_split_model)
        is_do_not_save_inst = (self.is_save_vocal_only and self.is_sec_bv_rebalance and stem_name == INST_STEM)

        if is_bv_rebalance_lead:
            master_voc_source = spec_utils.match_array_shapes(self.master_vocal_source, stem_source, is_swap=True)
            bv_rebalance_lead_source = stem_source-master_voc_source
            
        if not is_bv_model_lead and not is_do_not_save_inst:
            if self.is_vocal_split_model or not self.is_secondary_model:
                if self.is_vocal_split_model and not self.is_inst_only_voc_splitter:
                    save_voc_split_vocal(stem_name, stem_source)
                    if is_bv_rebalance_lead:
                        save_voc_split_vocal(LEAD_VOCAL_STEM, bv_rebalance_lead_source)
                else:
                    if not is_no_vocal_save:
                        save_with_message(stem_path, stem_name, stem_source)
                    
                if self.is_save_inst_vocal_splitter and not self.is_save_vocal_only:
                    save_voc_split_instrumental(stem_name, stem_source)
                    if is_bv_rebalance_lead:
                        save_voc_split_instrumental(LEAD_VOCAL_STEM, bv_rebalance_lead_source, is_inst_invert=True)

                self.set_progress_bar(0.95)

        if stem_name == VOCAL_STEM:
            self.master_vocal_path = stem_path
