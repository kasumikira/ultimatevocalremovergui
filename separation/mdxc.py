"""MDXC stem selection, separation and post-processing workflow."""

from __future__ import annotations

import os

import numpy as np
import soundfile as sf
import torch

from gui_data.constants import (
    ALL_STEMS,
    DEFAULT,
    DEMUD_COMBINE_METHODS,
    DEMUD_PHASE_INVERT,
    DEMUD_PHASE_ROTATE,
    DONE,
    INST_STEM,
    VOCAL_STEM,
    secondary_stem,
)
from lib_v5 import spec_utils
from lib_v5.verify_gpu_availability import clear_gpu_cache

from separation.audio import prepare_mix
from separation.inference.mdxc import MDXCInference
from separation.orchestration import process_secondary_model
from separation.inference.vr import vr_denoiser
from separation.workflow import SeperateAttributes

# MDXC backend: loading, chunk inference and stem processing

class SeperateMDXC(MDXCInference, SeperateAttributes):

    def seperate(self):
        self.is_vocal_main_target = True if self.mdx_c_configs.training.target_instrument == VOCAL_STEM else False
        samplerate = 44100
        sources = None
        if self.primary_model_name == self.model_basename and isinstance(self.primary_sources, tuple):
            mix, sources = self.primary_sources
            self.load_cached_sources()
        else:
            self.start_inference_console_write()
            self.running_inference_console_write()
            mix = prepare_mix(self.audio_file)
            sources = self.demix(mix)
            if not self.is_vocal_split_model:
                self.cache_source((mix, sources))
            self.write_to_console(DONE, base_text='')
        stem_list = [self.mdx_c_configs.training.target_instrument] if self.mdx_c_configs.training.target_instrument and not self.is_vocal_main_target else [i for i in self.mdx_c_configs.training.instruments]
        if self.is_secondary_model:
            if self.is_pre_proc_model:
                self.mdxnet_stem_select = stem_list[0]
            else:
                self.mdxnet_stem_select = self.main_model_primary_stem_4_stem if self.main_model_primary_stem_4_stem else self.primary_model_primary_stem
            if not self.is_vocal_split_model:
                self.primary_stem = self.mdxnet_stem_select
                self.secondary_stem = secondary_stem(self.mdxnet_stem_select)
            self.is_primary_stem_only, self.is_secondary_stem_only = False, False
        is_all_stems = self.mdxnet_stem_select == ALL_STEMS
        is_not_ensemble_master = not self.process_data['is_ensemble_master']
        is_not_single_stem = not len(stem_list) <= 2
        is_not_secondary_model = not self.is_secondary_model
        is_ensemble_4_stem = self.is_4_stem_ensemble and is_not_single_stem
        if (is_all_stems and is_not_ensemble_master and is_not_single_stem and is_not_secondary_model) or is_ensemble_4_stem and not self.is_pre_proc_model:
            print('going through stem list')
            for stem in stem_list:
                primary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({stem}).wav')
                self.primary_source = sources[stem].T
                self.write_audio(primary_stem_path, self.primary_source, samplerate, stem_name=stem)
                if stem == VOCAL_STEM and not self.is_sec_bv_rebalance:
                    self.process_vocal_split_chain({VOCAL_STEM:stem})
        else:
            is_temp_stem_swap = False
            if len(stem_list) == 1:
                source_primary = sources
            elif self.is_multi_stem_ensemble or len(stem_list) == 2:
                if self.is_secondary_model and self.mdxnet_stem_select in stem_list:
                    source_primary = sources[self.mdxnet_stem_select]
                else:
                    source_primary = sources[stem_list[0]]
            elif self.is_secondary_model and len(stem_list) > 2:
                secondary_stem_ = secondary_stem(self.mdxnet_stem_select)
                if secondary_stem_ in stem_list:
                    source_primary = sources[secondary_stem_]
                    self.primary_stem = secondary_stem_
                    is_temp_stem_swap = True
                else:
                    source_primary = sources[self.mdxnet_stem_select]
            else:
                source_primary = sources[self.mdxnet_stem_select]
            if self.is_secondary_model_activated and self.secondary_model:
                self.secondary_source_primary, self.secondary_source_secondary = process_secondary_model(self.secondary_model,
                                                                                                         self.process_data,
                                                                                                         main_process_method=self.process_method,
                                                                                                         main_model_primary=self.primary_stem)
            if not self.is_primary_stem_only:
                secondary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({self.secondary_stem}).wav')
                if not isinstance(self.secondary_source, np.ndarray):
                    if isinstance(sources, dict) and self.secondary_stem not in sources:
                        is_mdx_combine_stems = False
                    else:
                        is_mdx_combine_stems = self.is_mdx_combine_stems
                    if is_mdx_combine_stems and len(stem_list) >= 2:
                        if len(stem_list) == 2:
                            secondary_source = sources[self.secondary_stem]
                        else:
                            sources.pop(self.primary_stem)
                            next_stem = next(iter(sources))
                            secondary_source = np.zeros_like(sources[next_stem])
                            for v in sources.values():
                                secondary_source += v
                        if is_temp_stem_swap:
                            self.primary_stem, self.secondary_stem = secondary_stem(self.primary_model_primary_stem), self.primary_model_primary_stem
                        self.secondary_source = secondary_source.T
                    else:
                        self.secondary_source, raw_mix = source_primary, self.match_frequency_pitch(mix)
                        self.secondary_source = spec_utils.to_shape(self.secondary_source, raw_mix.shape)
                        if self.is_invert_spec:
                            self.secondary_source = spec_utils.invert_stem(raw_mix, self.secondary_source)
                        else:
                            self.secondary_source = (-self.secondary_source.T+raw_mix.T)
                self.secondary_source_map = self.final_process(secondary_stem_path, self.secondary_source, self.secondary_source_secondary, self.secondary_stem, samplerate)
            if not self.is_secondary_stem_only:
                primary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({self.primary_stem}).wav')
                if not isinstance(self.primary_source, np.ndarray):
                    self.primary_source = source_primary.T
                self.primary_source_map = self.final_process(primary_stem_path, self.primary_source, self.secondary_source_primary, self.primary_stem, samplerate)
        clear_gpu_cache()
        secondary_sources = {**self.primary_source_map, **self.secondary_source_map}
        self.process_vocal_split_chain(secondary_sources)
        if self.is_secondary_model or self.is_pre_proc_model:
            return secondary_sources


    def demix(self, mix, is_demud=False):
        sr_pitched = 441000
        org_mix = mix
        chunk_add = 3 if self.demudder_method == DEMUD_COMBINE_METHODS else 2
        if self.is_pitch_change:
            mix, sr_pitched = spec_utils.change_pitch_semitones(mix, 44100, semitone_shift=-self.semitone_shift)
        device = self.device
        model = self._load_model()
        mix = torch.tensor(mix, dtype=torch.float32, device=self.device)
        hop_size = self.find_hop_size(self.gen_model_config)
        is_seg_def = True if self.mdx_segment_size == DEFAULT else False
        if is_seg_def or self.is_force_mdx_c_seg_def:
            chunk_size = self.gen_model_config.audio.chunk_size
        else:
            chunk_size = hop_size * (self.mdx_segment_size - 1)
        is_target_inst = False
        num_instruments = 1 if self.gen_model_config.training.target_instrument else len(self.gen_model_config.training.instruments)
        num_overlap = self.overlap_mdx23
        target_stem = [self.mdx_c_configs.training.target_instrument] if self.mdx_c_configs.training.target_instrument else None
        if num_instruments == 1:
            is_target_inst = True if self.gen_model_config.training.target_instrument == INST_STEM else False
        estimated_sources = self._predict_chunks(
            model, mix, chunk_size, num_overlap, num_instruments, chunk_add,
        )
        pitch_fix = lambda s:self.pitch_fix(s, sr_pitched, org_mix)
        if num_instruments > 1 or self.is_vocal_main_target or is_target_inst:
            sources = {k: pitch_fix(v) if self.is_pitch_change else v for k, v in zip(target_stem if target_stem else self.mdx_c_configs.training.instruments, estimated_sources)}
            if self.is_vocal_main_target:
                if sources[VOCAL_STEM].shape[1] != org_mix.shape[1]:
                    sources[VOCAL_STEM] = spec_utils.match_array_shapes(sources[VOCAL_STEM], org_mix)
                sources[INST_STEM] = org_mix - sources[VOCAL_STEM]
            if is_target_inst:
                if sources[INST_STEM].shape[1] != org_mix.shape[1]:
                    sources[INST_STEM] = spec_utils.match_array_shapes(sources[INST_STEM], org_mix)
                sources[VOCAL_STEM] = org_mix - sources[INST_STEM]
            if self.is_denoise_model and VOCAL_STEM in sources.keys() and INST_STEM in sources.keys():
                sources[VOCAL_STEM] = vr_denoiser(sources[VOCAL_STEM], self.device, model_path=self.DENOISER_MODEL)
                if sources[VOCAL_STEM].shape[1] != org_mix.shape[1]:
                    sources[VOCAL_STEM] = spec_utils.match_array_shapes(sources[VOCAL_STEM], org_mix)
                sources[INST_STEM] = org_mix - sources[VOCAL_STEM]
            if is_demud:
                return sources[VOCAL_STEM]
            elif VOCAL_STEM in sources.keys() and INST_STEM in sources.keys() and self.is_demud:
                self.write_to_console('De-mudding Instrumental stem... ', base_text='')
                inst_source = spec_utils.match_array_shapes(sources[INST_STEM], org_mix)
                if self.demudder_method == DEMUD_COMBINE_METHODS:
                    bare_source = org_mix - inst_source
                    phase_app_mix = spec_utils.demud_processor(org_mix, inst_source, demudder_method=DEMUD_PHASE_ROTATE)
                    phase_app_inv_mix = spec_utils.demud_processor(org_mix, inst_source, demudder_method=DEMUD_PHASE_INVERT)
                    phase_stem_remix = self.demix(phase_app_mix, is_demud=True)
                    phase_stem_inv = self.demix(phase_app_inv_mix, is_demud=True)
                    bare_stem_list = [bare_source, phase_stem_remix, phase_stem_inv]
                    phase_stem = spec_utils.average_audio(bare_stem_list, is_demud=True)
                else:
                    phase_app_mix = spec_utils.demud_processor(org_mix, inst_source, demudder_method=self.demudder_method)
                    phase_stem = self.demix(phase_app_mix, is_demud=True)
                try:
                    sf.write(os.path.join('demud_tests', f'{self.audio_file_base}_(phased_mix).wav'), phase_app_mix.T, 44100, subtype=self.wav_type_set)
                except:
                    print('Failed to save inverted file')
                sources[INST_STEM] = org_mix - spec_utils.match_array_shapes(phase_stem, org_mix)
            if is_target_inst and VOCAL_STEM in sources.keys():
                sources = sources[INST_STEM]
            return sources
        else:
            sources = {k: v for k, v in zip([self.mdx_c_configs.training.target_instrument], estimated_sources)}
            est_s = sources[self.mdx_c_configs.training.target_instrument]
            return pitch_fix(est_s) if self.is_pitch_change else est_s
