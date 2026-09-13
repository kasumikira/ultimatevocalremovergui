"""MDX model loading and separation workflow."""

from __future__ import annotations

import os

import numpy as np
import onnxruntime as ort
import torch
from onnx import load
from onnx2pytorch import ConvertModel

import lib_v5.mdxnet as MdxnetSet
from gui_data.constants import (
    DEFAULT,
    DONE,
    MDX_NET_FREQ_CUT,
    PRIMARY_STEM,
    SECONDARY_STEM,
)
from lib_v5 import spec_utils
from lib_v5.verify_gpu_availability import clear_gpu_cache, onnxruntime_cuda_available
from separation.audio import prepare_mix
from separation.inference.mdx import MDXInference
from separation.orchestration import process_secondary_model
from separation.workflow import SeperateAttributes

# MDX backend

class SeperateMDX(MDXInference, SeperateAttributes):        

    def seperate(self):
        samplerate = 44100
    
        if self.mdx_segment_size == DEFAULT:
            self.mdx_segment_size = self.dim_t
    
        if self.primary_model_name == self.model_basename and isinstance(self.primary_sources, tuple):
            mix, source = self.primary_sources
            self.load_cached_sources()
        else:
            self.start_inference_console_write()

            if self.is_mdx_ckpt:
                model_params = torch.load(self.model_path, map_location=lambda storage, loc: storage)['hyper_parameters']
                self.dim_c, self.hop = model_params['dim_c'], model_params['hop_length']
                separator = MdxnetSet.ConvTDFNet(**model_params)
                self.model_run = separator.load_from_checkpoint(self.model_path).to(self.device).eval()
            else:
                if self.mdx_segment_size == self.dim_t and onnxruntime_cuda_available:
                    ort_ = ort.InferenceSession(self.model_path, providers=self.run_type)
                    self.model_run = lambda spek:ort_.run(None, {'input': spek.cpu().numpy()})[0]
                else:
                    self.model_run = ConvertModel(load(self.model_path))
                    self.model_run.to(self.device).eval()

            self.running_inference_console_write()
            mix = prepare_mix(self.audio_file)
            
            source = self.demix(mix)
            
            mdx_net_cut = True if self.primary_stem in MDX_NET_FREQ_CUT and self.is_match_frequency_pitch else False

            if not isinstance(source, np.ndarray) and type(source) is dict:
                self.secondary_source = self.demix(source[SECONDARY_STEM], is_match_mix=True).T if mdx_net_cut else source[SECONDARY_STEM]
                source = source[PRIMARY_STEM]
            
            if not self.is_vocal_split_model:
                self.cache_source((mix, source))
            self.write_to_console(DONE, base_text='')            

        if self.is_secondary_model_activated and self.secondary_model:
            self.secondary_source_primary, self.secondary_source_secondary = process_secondary_model(self.secondary_model, self.process_data, main_process_method=self.process_method, main_model_primary=self.primary_stem)
        
        if not self.is_primary_stem_only:
            secondary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({self.secondary_stem}).wav')
            if not isinstance(self.secondary_source, np.ndarray):
                raw_mix = self.demix(self.match_frequency_pitch(mix), is_match_mix=True) if mdx_net_cut else self.match_frequency_pitch(mix)
                self.secondary_source = spec_utils.invert_stem(raw_mix, source) if self.is_invert_spec else raw_mix.T-source.T
            
            self.secondary_source_map = self.final_process(secondary_stem_path, self.secondary_source, self.secondary_source_secondary, self.secondary_stem, samplerate)
        
        if not self.is_secondary_stem_only:
            primary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({self.primary_stem}).wav')

            if not isinstance(self.primary_source, np.ndarray):
                self.primary_source = source.T
                
            self.primary_source_map = self.final_process(primary_stem_path, self.primary_source, self.secondary_source_primary, self.primary_stem, samplerate)
        
        clear_gpu_cache()

        secondary_sources = {**self.primary_source_map, **self.secondary_source_map}
        
        self.process_vocal_split_chain(secondary_sources)

        if self.is_secondary_model or self.is_pre_proc_model:
            return secondary_sources
