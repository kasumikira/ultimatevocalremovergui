from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import numpy as np
import torch

from gui_data.constants import (
    DEMUCS_ARCH_TYPE,
    DEMUCS_V3,
    DEMUCS_V4,
    INST_STEM,
    MDX_ARCH_TYPE,
    VOCAL_STEM,
    VR_ARCH_TYPE,
    secondary_stem,
)
from lib_v5.verify_gpu_availability import (
    GPU_TYPE_APPLE_MPS,
    GPU_TYPE_CPU,
    check_gpu_availability,
    onnxruntime_cuda_available,
)

if TYPE_CHECKING:
    from UVR import ModelData

cpu = torch.device('cpu')

class SeparationState:
    def __init__(self, model_data: ModelData, 
                 process_data: dict, 
                 main_model_primary_stem_4_stem=None, 
                 main_process_method=None, 
                 is_return_dual=True, 
                 main_model_primary=None, 
                 vocal_stem_path=None, 
                 master_inst_source=None,
                 master_vocal_source=None):
        
        self.list_all_models: list
        self.process_data = process_data
        self.progress_value = 0
        self.set_progress_bar = process_data['set_progress_bar']
        self.write_to_console = process_data['write_to_console']
        if vocal_stem_path:
            self.audio_file, self.audio_file_base = vocal_stem_path
            self.audio_file_base_voc_split = lambda stem, split:os.path.join(self.export_path, f'{self.audio_file_base.replace("_(Vocals)", "")}_({stem}_{split}).wav')
        else:
            self.audio_file = process_data['audio_file']
            self.audio_file_base = process_data['audio_file_base']
            self.audio_file_base_voc_split = None
        self.export_path = process_data['export_path']
        self.cached_source_callback = process_data['cached_source_callback']
        self.cached_model_source_holder = process_data['cached_model_source_holder']
        self.is_4_stem_ensemble = process_data['is_4_stem_ensemble']
        self.list_all_models = process_data['list_all_models']
        self.process_iteration = process_data['process_iteration']
        self.is_return_dual = is_return_dual
        self.is_pitch_change = model_data.is_pitch_change
        self.semitone_shift = model_data.semitone_shift
        self.is_match_frequency_pitch = model_data.is_match_frequency_pitch
        self.overlap = model_data.overlap
        self.overlap_mdx = model_data.overlap_mdx
        self.overlap_mdx23 = model_data.overlap_mdx23
        self.is_mdx_combine_stems = model_data.is_mdx_combine_stems
        self.is_mdx_c = model_data.is_mdx_c
        self.mdx_c_configs = model_data.mdx_c_configs
        self.mdxnet_stem_select = model_data.mdxnet_stem_select
        self.mixer_path = model_data.mixer_path
        self.model_samplerate = model_data.model_samplerate
        self.model_capacity = model_data.model_capacity
        self.is_vr_51_model = model_data.is_vr_51_model
        self.is_pre_proc_model = model_data.is_pre_proc_model
        self.is_secondary_model_activated = model_data.is_secondary_model_activated if not self.is_pre_proc_model else False
        self.is_secondary_model = model_data.is_secondary_model if not self.is_pre_proc_model else True
        self.process_method = model_data.process_method
        self.model_path = model_data.model_path
        self.model_name = model_data.model_name
        self.model_basename = model_data.model_basename
        self.wav_type_set = model_data.wav_type_set
        self.mp3_bit_set = model_data.mp3_bit_set
        self.save_format = model_data.save_format
        self.is_gpu_conversion = model_data.is_gpu_conversion
        self.is_normalization = model_data.is_normalization
        self.is_primary_stem_only = model_data.is_primary_stem_only if not self.is_secondary_model else model_data.is_primary_model_primary_stem_only
        self.is_secondary_stem_only = model_data.is_secondary_stem_only if not self.is_secondary_model else model_data.is_primary_model_secondary_stem_only      
        self.is_ensemble_mode = model_data.is_ensemble_mode
        self.secondary_model = model_data.secondary_model #
        self.primary_model_primary_stem = model_data.primary_model_primary_stem
        self.primary_stem_native = model_data.primary_stem_native
        self.primary_stem = model_data.primary_stem #
        # print(self.primary_stem)
        self.secondary_stem = model_data.secondary_stem #
        self.is_invert_spec = model_data.is_invert_spec #
        self.is_deverb_vocals = model_data.is_deverb_vocals
        self.is_target_instrument = model_data.is_target_instrument
        self.is_mixer_mode = model_data.is_mixer_mode #
        self.secondary_model_scale = model_data.secondary_model_scale #
        self.is_demucs_pre_proc_model_inst_mix = model_data.is_demucs_pre_proc_model_inst_mix #
        self.primary_source_map = {}
        self.secondary_source_map = {}
        self.primary_source = None
        self.secondary_source = None
        self.secondary_source_primary = None
        self.secondary_source_secondary = None
        self.main_model_primary_stem_4_stem = main_model_primary_stem_4_stem
        self.main_model_primary = main_model_primary
        self.ensemble_primary_stem = model_data.ensemble_primary_stem
        self.is_multi_stem_ensemble = model_data.is_multi_stem_ensemble
        self.gpu_type = GPU_TYPE_CPU
        self.is_deverb = True
        self.DENOISER_MODEL = model_data.DENOISER_MODEL
        self.DEVERBER_MODEL = model_data.DEVERBER_MODEL
        self.is_source_swap = False
        self.vocal_split_model = model_data.vocal_split_model
        self.is_vocal_split_model = model_data.is_vocal_split_model
        self.master_vocal_path = None
        self.set_master_inst_source = None
        self.master_inst_source = master_inst_source
        self.master_vocal_source = master_vocal_source
        self.is_save_inst_vocal_splitter = isinstance(master_inst_source, np.ndarray) and model_data.is_save_inst_vocal_splitter
        self.is_inst_only_voc_splitter = model_data.is_inst_only_voc_splitter
        self.is_karaoke = model_data.is_karaoke
        self.is_bv_model = model_data.is_bv_model
        self.is_bv_model_rebalenced = model_data.bv_model_rebalance and self.is_vocal_split_model
        self.is_sec_bv_rebalance = model_data.is_sec_bv_rebalance
        self.stem_path_init = os.path.join(self.export_path, f'{self.audio_file_base}_({self.secondary_stem}).wav')
        self.deverb_vocal_opt = model_data.deverb_vocal_opt
        self.is_save_vocal_only = model_data.is_save_vocal_only
        self.device = cpu
        self.run_type = ['CPUExecutionProvider']
        self.device_set = model_data.device_set
        self.is_demud = model_data.is_demud
        self.demudder_method = model_data.demudder_method
        self.gen_model_config = model_data.mdx_c_configs
        self.is_roformer = model_data.is_roformer
        self.overwrite_protect_stamp = round(time.time())
        self.mdx_model_type = model_data.mdx_model_type
        self.is_force_mdx_c_seg_def = model_data.is_force_mdx_c_seg_def
        self.is_use_torch_inference_mode = model_data.is_use_torch_inference_mode
        self.is_calculate_comp = model_data.is_calculate_comp

        if self.is_inst_only_voc_splitter or self.is_sec_bv_rebalance:
            self.is_primary_stem_only = False
            self.is_secondary_stem_only = False
        
        if main_model_primary and self.is_multi_stem_ensemble:
            self.primary_stem, self.secondary_stem = main_model_primary, secondary_stem(main_model_primary)

        self.device, self.gpu_type = check_gpu_availability(
            self.is_gpu_conversion,
            self.device_set,
        )
        if onnxruntime_cuda_available:
            self.run_type = ['CUDAExecutionProvider']

        if model_data.process_method == MDX_ARCH_TYPE:
            self.is_mdx_ckpt = model_data.is_mdx_ckpt
            self.primary_model_name, self.primary_sources = self.cached_source_callback(MDX_ARCH_TYPE, model_name=self.model_basename)
            self.is_denoise = model_data.is_denoise#
            self.is_denoise_model = model_data.is_denoise_model#
            self.mdx_batch_size = model_data.mdx_batch_size
            self.compensate = model_data.compensate
            self.mdx_segment_size = model_data.mdx_segment_size
            
            if self.is_mdx_c:
                if not self.is_4_stem_ensemble:
                    if not self.is_target_instrument:
                        self.primary_stem = model_data.ensemble_primary_stem if process_data['is_ensemble_master'] else model_data.primary_stem
                        self.secondary_stem = model_data.ensemble_secondary_stem if process_data['is_ensemble_master'] else model_data.secondary_stem
            else:
                self.dim_f, self.dim_t = model_data.mdx_dim_f_set, 2**model_data.mdx_dim_t_set
                
            self.check_label_secondary_stem_runs()
            self.n_fft = model_data.mdx_n_fft_scale_set
            self.chunks = model_data.chunks
            self.margin = model_data.margin
            self.adjust = 1
            self.dim_c = 4
            self.hop = 1024

        if model_data.process_method == DEMUCS_ARCH_TYPE:
            self.demucs_stems = model_data.demucs_stems if not main_process_method in [MDX_ARCH_TYPE, VR_ARCH_TYPE] else None
            self.secondary_model_4_stem = model_data.secondary_model_4_stem
            self.secondary_model_4_stem_scale = model_data.secondary_model_4_stem_scale
            self.is_chunk_demucs = model_data.is_chunk_demucs
            self.segment = model_data.segment
            self.demucs_version = model_data.demucs_version
            self.demucs_source_list = model_data.demucs_source_list
            self.demucs_source_map = model_data.demucs_source_map
            self.is_custom_demucs = model_data.is_custom_demucs
            self.is_demucs_combine_stems = model_data.is_demucs_combine_stems
            self.demucs_stem_count = model_data.demucs_stem_count
            self.pre_proc_model = model_data.pre_proc_model
            self.device = cpu if self.gpu_type == GPU_TYPE_APPLE_MPS and not self.demucs_version in [DEMUCS_V3, DEMUCS_V4] else self.device

            self.primary_stem = model_data.ensemble_primary_stem if process_data['is_ensemble_master'] else model_data.primary_stem
            self.secondary_stem = model_data.ensemble_secondary_stem if process_data['is_ensemble_master'] else model_data.secondary_stem

            if (self.is_multi_stem_ensemble or self.is_4_stem_ensemble) and not self.is_secondary_model:
                self.is_return_dual = False
            
            if self.is_multi_stem_ensemble and main_model_primary:
                self.is_4_stem_ensemble = False
                if main_model_primary in self.demucs_source_map.keys():
                    self.primary_stem = main_model_primary
                    self.secondary_stem = secondary_stem(main_model_primary)
                elif secondary_stem(main_model_primary) in self.demucs_source_map.keys():
                    self.primary_stem = secondary_stem(main_model_primary)
                    self.secondary_stem = main_model_primary

            if self.is_secondary_model and not process_data['is_ensemble_master']:
                if not self.demucs_stem_count == 2 and model_data.primary_model_primary_stem == INST_STEM:
                    self.primary_stem = VOCAL_STEM
                    self.secondary_stem = INST_STEM
                else:
                    self.primary_stem = model_data.primary_model_primary_stem
                    self.secondary_stem = secondary_stem(self.primary_stem)

            self.shifts = model_data.shifts
            self.is_split_mode = model_data.is_split_mode if not self.demucs_version == DEMUCS_V4 else True
            self.primary_model_name, self.primary_sources = self.cached_source_callback(DEMUCS_ARCH_TYPE, model_name=self.model_basename)

        if model_data.process_method == VR_ARCH_TYPE:
            self.check_label_secondary_stem_runs()
            self.primary_model_name, self.primary_sources = self.cached_source_callback(VR_ARCH_TYPE, model_name=self.model_basename)
            self.mp = model_data.vr_model_param
            self.high_end_process = model_data.is_high_end_process
            self.is_tta = model_data.is_tta
            self.is_post_process = model_data.is_post_process
            self.is_gpu_conversion = model_data.is_gpu_conversion
            self.batch_size = model_data.batch_size
            self.window_size = model_data.window_size
            self.input_high_end_h = None
            self.input_high_end = None
            self.post_process_threshold = model_data.post_process_threshold
            self.aggressiveness = {'value': model_data.aggression_setting, 
                                   'split_bin': self.mp.param['band'][1]['crop_stop'], 
                                   'aggr_correction': self.mp.param.get('aggr_correction')}
