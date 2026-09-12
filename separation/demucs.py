"""Demucs model loading and separation workflow."""

from __future__ import annotations

from pathlib import Path
import gzip
import os

import numpy as np
import torch

from demucs.apply import demucs_segments
from demucs.hdemucs import HDemucs
from demucs.model_v2 import auto_load_demucs_model_v2
from demucs.pretrained import get_model as _gm
from gui_data.constants import (
    ALL_STEMS,
    DEMUCS_2_SOURCE_MAPPER,
    DEMUCS_4_SOURCE_MAPPER,
    DEMUCS_6_SOURCE_MAPPER,
    DEMUCS_V1,
    DEMUCS_V2,
    DONE,
    GUITAR_STEM,
    INST_STEM,
    OTHER_STEM,
    PIANO_STEM,
    VOCAL_STEM,
)
from lib_v5 import spec_utils
from lib_v5.verify_gpu_availability import clear_gpu_cache

from separation.audio import prepare_mix
from separation.inference.demucs import DemucsInference
from separation.orchestration import process_secondary_model
from separation.workflow import SeperateAttributes

# Demucs backend

class SeperateDemucs(DemucsInference, SeperateAttributes):
    def seperate(self):
        samplerate = 44100
        source = None
        model_scale = None
        stem_source = None
        stem_source_secondary = None
        inst_mix = None
        inst_source = None
        is_no_write = False
        is_no_piano_guitar = False
        is_no_cache = False
        
        if self.primary_model_name == self.model_basename and isinstance(self.primary_sources, np.ndarray) and not self.pre_proc_model:
            source = self.primary_sources
            self.load_cached_sources()
        else:
            self.start_inference_console_write()
            is_no_cache = True

        mix = prepare_mix(self.audio_file)

        if is_no_cache:
            if self.demucs_version == DEMUCS_V1:
                if str(self.model_path).endswith(".gz"):
                    self.model_path = gzip.open(self.model_path, "rb")
                klass, args, kwargs, state = torch.load(self.model_path)
                self.demucs = klass(*args, **kwargs)
                self.demucs.to(self.device) 
                self.demucs.load_state_dict(state)
            elif self.demucs_version == DEMUCS_V2:
                self.demucs = auto_load_demucs_model_v2(self.demucs_source_list, self.model_path)
                self.demucs.to(self.device) 
                self.demucs.load_state_dict(torch.load(self.model_path))
                self.demucs.eval()
            else:  
                print('demucs_source_list', self.demucs_source_list)
                print('model_path', self.model_path)
                print('segment', self.segment)
                self.demucs = HDemucs(sources=self.demucs_source_list)
                self.demucs = _gm(name=os.path.splitext(os.path.basename(self.model_path))[0], 
                                  repo=Path(os.path.dirname(self.model_path)))
                self.demucs = demucs_segments(self.segment, self.demucs)
                self.demucs.to(self.device)
                self.demucs.eval()

            if self.pre_proc_model:
                if self.primary_stem not in [VOCAL_STEM, INST_STEM]:
                    is_no_write = True
                    self.write_to_console(DONE, base_text='')
                    mix_no_voc = process_secondary_model(self.pre_proc_model, self.process_data, is_pre_proc_model=True)
                    inst_mix = prepare_mix(mix_no_voc[INST_STEM])
                    self.process_iteration()
                    self.running_inference_console_write(is_no_write=is_no_write)
                    inst_source = self.demix_demucs(inst_mix)
                    self.process_iteration()

            self.running_inference_console_write(is_no_write=is_no_write) if not self.pre_proc_model else None
            
            if self.primary_model_name == self.model_basename and isinstance(self.primary_sources, np.ndarray) and self.pre_proc_model:
                source = self.primary_sources
            else:
                source = self.demix_demucs(mix)
            
            self.write_to_console(DONE, base_text='')
            
            del self.demucs
            clear_gpu_cache()
            
        if isinstance(inst_source, np.ndarray):
            source_reshape = spec_utils.reshape_sources(inst_source[self.demucs_source_map[VOCAL_STEM]], source[self.demucs_source_map[VOCAL_STEM]])
            inst_source[self.demucs_source_map[VOCAL_STEM]] = source_reshape
            source = inst_source

        if isinstance(source, np.ndarray):
            
            if not self.is_custom_demucs:
                if len(source) == 2:
                    self.demucs_source_map = DEMUCS_2_SOURCE_MAPPER
                else:
                    self.demucs_source_map = DEMUCS_6_SOURCE_MAPPER if len(source) == 6 else DEMUCS_4_SOURCE_MAPPER

                    if len(source) == 6 and self.process_data['is_ensemble_master'] or len(source) == 6 and self.is_secondary_model:
                        is_no_piano_guitar = True
                        six_stem_other_source = list(source)
                        six_stem_other_source = [i for n, i in enumerate(source) if n in [self.demucs_source_map[OTHER_STEM], self.demucs_source_map[GUITAR_STEM], self.demucs_source_map[PIANO_STEM]]]
                        other_source = np.zeros_like(six_stem_other_source[0])
                        for i in six_stem_other_source:
                            other_source += i
                        source_reshape = spec_utils.reshape_sources(source[self.demucs_source_map[OTHER_STEM]], other_source)
                        source[self.demucs_source_map[OTHER_STEM]] = source_reshape
                    
        if not self.is_vocal_split_model:
            self.cache_source(source)
        
        if (self.demucs_stems == ALL_STEMS and not self.process_data['is_ensemble_master']) or self.is_4_stem_ensemble and not self.is_return_dual:
            for stem_name, stem_value in self.demucs_source_map.items():
                if self.is_secondary_model_activated and not self.is_secondary_model and not stem_value >= 4:
                    if self.secondary_model_4_stem[stem_value]:
                        model_scale = self.secondary_model_4_stem_scale[stem_value]
                        stem_source_secondary = process_secondary_model(self.secondary_model_4_stem[stem_value], self.process_data, main_model_primary_stem_4_stem=stem_name, is_source_load=True, is_return_dual=False)
                        if isinstance(stem_source_secondary, np.ndarray):
                            stem_source_secondary = stem_source_secondary[1 if self.secondary_model_4_stem[stem_value].demucs_stem_count == 2 else stem_value].T
                        elif type(stem_source_secondary) is dict:
                            stem_source_secondary = stem_source_secondary[stem_name]
                            
                stem_source_secondary = None if stem_value >= 4 else stem_source_secondary
                stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({stem_name}).wav')
                stem_source = source[stem_value].T
                
                stem_source = self.process_secondary_stem(stem_source, secondary_model_source=stem_source_secondary, model_scale=model_scale)
                self.write_audio(stem_path, stem_source, samplerate, stem_name=stem_name)
                
                if stem_name == VOCAL_STEM and not self.is_sec_bv_rebalance:
                    self.process_vocal_split_chain({VOCAL_STEM:stem_source})
                
            if self.is_secondary_model:    
                return source
        else:
            if self.is_secondary_model_activated and self.secondary_model:
                    self.secondary_source_primary, self.secondary_source_secondary = process_secondary_model(self.secondary_model, self.process_data, main_process_method=self.process_method)
                    
            if not self.is_primary_stem_only:
                def secondary_save(sec_stem_name, source, raw_mixture=None, is_inst_mixture=False):
                    secondary_source = self.secondary_source if not is_inst_mixture else None
                    secondary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({sec_stem_name}).wav')
                    secondary_source_secondary = None
                    
                    if not isinstance(secondary_source, np.ndarray):
                        if self.is_demucs_combine_stems:
                            source = list(source)
                            if is_inst_mixture:
                                source = [i for n, i in enumerate(source) if not n in [self.demucs_source_map[self.primary_stem], self.demucs_source_map[VOCAL_STEM]]]
                            else:
                                source.pop(self.demucs_source_map[self.primary_stem])
                                
                            source = source[:len(source) - 2] if is_no_piano_guitar else source
                            secondary_source = np.zeros_like(source[0])
                            for i in source:
                                secondary_source += i
                            secondary_source = secondary_source.T
                        else:
                            if not isinstance(raw_mixture, np.ndarray):
                                raw_mixture = prepare_mix(self.audio_file)
       
                            secondary_source = source[self.demucs_source_map[self.primary_stem]]
                            
                            if self.is_invert_spec:
                                secondary_source = spec_utils.invert_stem(raw_mixture, secondary_source)
                            else:
                                raw_mixture = spec_utils.reshape_sources(secondary_source, raw_mixture)
                                secondary_source = (-secondary_source.T+raw_mixture.T)
                            
                    if not is_inst_mixture:
                        self.secondary_source = secondary_source
                        secondary_source_secondary = self.secondary_source_secondary
                        self.secondary_source = self.process_secondary_stem(secondary_source, secondary_source_secondary)
                        self.secondary_source_map = {self.secondary_stem: self.secondary_source}

                    self.write_audio(secondary_stem_path, secondary_source, samplerate, stem_name=sec_stem_name)

                secondary_save(self.secondary_stem, source, raw_mixture=mix)
                
                if self.is_demucs_pre_proc_model_inst_mix and self.pre_proc_model and not self.is_4_stem_ensemble:
                    secondary_save(f"{self.secondary_stem} {INST_STEM}", source, raw_mixture=inst_mix, is_inst_mixture=True)

            if not self.is_secondary_stem_only:
                primary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({self.primary_stem}).wav')
                if not isinstance(self.primary_source, np.ndarray):
                    self.primary_source = source[self.demucs_source_map[self.primary_stem]].T
                
                self.primary_source_map = self.final_process(primary_stem_path, self.primary_source, self.secondary_source_primary, self.primary_stem, samplerate)

            secondary_sources = {**self.primary_source_map, **self.secondary_source_map}
            
            self.process_vocal_split_chain(secondary_sources)
            
            if self.is_secondary_model:    
                return secondary_sources
