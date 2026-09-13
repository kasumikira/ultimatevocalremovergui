"""VR model loading and separation workflow."""

from __future__ import annotations

import math
import os

import librosa
import numpy as np
import torch

from gui_data.constants import DONE
from lib_v5.verify_gpu_availability import clear_gpu_cache
from lib_v5.vr_network import nets, nets_new
from separation.inference.vr import VRInference
from separation.orchestration import process_secondary_model
from separation.workflow import SeperateAttributes

cpu = torch.device('cpu')

# VR backend
class SeperateVR(VRInference, SeperateAttributes):        
    def seperate(self):
        if self.primary_model_name == self.model_basename and isinstance(self.primary_sources, tuple):
            y_spec, v_spec = self.primary_sources
            self.load_cached_sources()
        else:
            self.start_inference_console_write()

            device = self.device

            nn_arch_sizes = [
                31191, # default
                33966, 56817, 123821, 123812, 129605, 218409, 537238, 537227]
            vr_5_1_models = [56817, 218409]
            model_size = math.ceil(os.stat(self.model_path).st_size / 1024)
            nn_arch_size = min(nn_arch_sizes, key=lambda x:abs(x-model_size))

            if nn_arch_size in vr_5_1_models or self.is_vr_51_model:
                self.model_run = nets_new.CascadedNet(self.mp.param['bins'] * 2, 
                                                      nn_arch_size, 
                                                      nout=self.model_capacity[0], 
                                                      nout_lstm=self.model_capacity[1])
                self.is_vr_51_model = True
            else:
                self.model_run = nets.determine_model_capacity(self.mp.param['bins'] * 2, nn_arch_size)
                            
            self.model_run.load_state_dict(torch.load(self.model_path, map_location=cpu)) 
            self.model_run.to(device) 

            self.running_inference_console_write()
                        
            y_spec, v_spec = self.inference_vr(self.loading_mix(), device, self.aggressiveness)
            if not self.is_vocal_split_model:
                self.cache_source((y_spec, v_spec))
            self.write_to_console(DONE, base_text='')
            
        if self.is_secondary_model_activated and self.secondary_model:
            self.secondary_source_primary, self.secondary_source_secondary = process_secondary_model(self.secondary_model, self.process_data, main_process_method=self.process_method, main_model_primary=self.primary_stem)

        if not self.is_secondary_stem_only:
            primary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({self.primary_stem}).wav')
            if not isinstance(self.primary_source, np.ndarray):
                self.primary_source = self.spec_to_wav(y_spec).T
                if not self.model_samplerate == 44100:
                    self.primary_source = librosa.resample(self.primary_source.T, orig_sr=self.model_samplerate, target_sr=44100).T
                
            self.primary_source_map = self.final_process(primary_stem_path, self.primary_source, self.secondary_source_primary, self.primary_stem, 44100)  

        if not self.is_primary_stem_only:
            secondary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({self.secondary_stem}).wav')
            if not isinstance(self.secondary_source, np.ndarray):
                self.secondary_source = self.spec_to_wav(v_spec).T
                if not self.model_samplerate == 44100:
                    self.secondary_source = librosa.resample(self.secondary_source.T, orig_sr=self.model_samplerate, target_sr=44100).T
            
            self.secondary_source_map = self.final_process(secondary_stem_path, self.secondary_source, self.secondary_source_secondary, self.secondary_stem, 44100)
            
        clear_gpu_cache()
        secondary_sources = {**self.primary_source_map, **self.secondary_source_map}
        
        self.process_vocal_split_chain(secondary_sources)
        
        if self.is_secondary_model:
            return secondary_sources
            
