from __future__ import annotations

import numpy as np
import torch

from .vr import vr_denoiser
from gui_data.constants import (
    DEFAULT,
    DEMUD_COMBINE_METHODS,
    DEMUD_PHASE_INVERT,
    DEMUD_PHASE_ROTATE,
    INST_STEM,
    NO_STEM,
    PRIMARY_STEM,
    SECONDARY_STEM,
)
from lib_v5 import spec_utils
from lib_v5.tfc_tdf_v3 import STFT


class MDXInference:
    def initialize_model_settings(self):
        self.n_bins = self.n_fft//2+1
        self.trim = self.n_fft//2
        self.chunk_size = self.hop * (self.mdx_segment_size-1)
        self.gen_size = self.chunk_size-2*self.trim
        self.stft = STFT(self.n_fft, self.hop, self.dim_f, self.device)


    def demix(self, mix, is_match_mix=False, is_demud=False):
        self.initialize_model_settings()
        
        is_bare_stem = NO_STEM not in self.primary_stem_native and not self.primary_stem_native == INST_STEM
        org_mix = mix
        tar_waves_ = []

        comp_valu = self.compensate
        is_calculate_comp = False if is_match_mix or is_bare_stem else self.is_calculate_comp
        chunk_add = 3 if self.demudder_method == DEMUD_COMBINE_METHODS else 2
        if is_match_mix:
            chunk_size = self.hop * (256-1)
            overlap = 0.02
        else:
            chunk_size = self.chunk_size
            overlap = self.overlap_mdx
            
            if self.is_pitch_change:
                mix, sr_pitched = spec_utils.change_pitch_semitones(mix, 44100, semitone_shift=-self.semitone_shift)

        gen_size = chunk_size-2*self.trim

        pad = gen_size + self.trim - ((mix.shape[-1]) % gen_size)
        mixture = np.concatenate((np.zeros((2, self.trim), dtype='float32'), mix, np.zeros((2, pad), dtype='float32')), 1)

        step = self.chunk_size - self.n_fft if overlap == DEFAULT else int((1 - overlap) * chunk_size)
        result = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
        divider = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
        total = 0
        total_chunks = (mixture.shape[-1] + step - 1) // step

        if self.is_demud:
            total_chunks = total_chunks * chunk_add
            print('Total Chunks: ', total_chunks)

        for i in range(0, mixture.shape[-1], step):
            total += 1
            start = i
            end = min(i + chunk_size, mixture.shape[-1])

            chunk_size_actual = end - start

            if overlap == 0:
                window = None
            else:
                window = np.hanning(chunk_size_actual)
                window = np.tile(window[None, None, :], (1, 2, 1))

            mix_part_ = mixture[:, start:end]
            if end != i + chunk_size:
                pad_size = (i + chunk_size) - end
                mix_part_ = np.concatenate((mix_part_, np.zeros((2, pad_size), dtype='float32')), axis=-1)

            mix_part = torch.tensor([mix_part_], dtype=torch.float32).to(self.device)
            mix_waves = mix_part.split(1)
            
            with torch.no_grad():
                for mix_wave in mix_waves:
                    self.running_inference_progress_bar(total_chunks, is_match_mix=is_match_mix)

                    tar_waves = self.run_model(mix_wave, is_match_mix=is_match_mix)
                    
                    if window is not None:
                        tar_waves[..., :chunk_size_actual] *= window 
                        divider[..., start:end] += window
                    else:
                        divider[..., start:end] += 1

                    result[..., start:end] += tar_waves[..., :end-start]
            
        tar_waves = result / divider
        tar_waves_.append(tar_waves)

        tar_waves_ = np.vstack(tar_waves_)[:, :, self.trim:-self.trim]
        tar_waves = np.concatenate(tar_waves_, axis=-1)[:, :mix.shape[-1]]
        
        source = tar_waves[:,0:None]

        if self.is_pitch_change and not is_match_mix:
            source = self.pitch_fix(source, sr_pitched, org_mix)

        if is_calculate_comp:
            comp_valu = spec_utils.calculate_comp_level(spec_utils.match_array_shapes(source, org_mix), org_mix, comp_set=self.compensate)
        source = source if is_match_mix else source*comp_valu

        if self.is_denoise_model and not is_match_mix:
            if NO_STEM in self.primary_stem_native or self.primary_stem_native == INST_STEM:
                if org_mix.shape[1] != source.shape[1]:
                    source = spec_utils.match_array_shapes(source, org_mix)
                source = org_mix - vr_denoiser(org_mix-source, self.device, model_path=self.DENOISER_MODEL)
            else:
                source = vr_denoiser(source, self.device, model_path=self.DENOISER_MODEL)

        if is_demud:
            source = spec_utils.match_array_shapes(source, org_mix)
            source = source if is_bare_stem else org_mix - source
            return source

        if self.is_demud and not is_match_mix and not self.is_vocal_split_model:
            self.write_to_console("De-mudding Instrumental stem... ", base_text="")
            inst_source = org_mix - spec_utils.match_array_shapes(source, org_mix) if is_bare_stem else source
            inst_source = spec_utils.match_array_shapes(inst_source, org_mix)
            bare_source = org_mix - inst_source
            if self.demudder_method == DEMUD_COMBINE_METHODS:
                phase_app_mix = spec_utils.demud_processor(
                    org_mix, inst_source, demudder_method=DEMUD_PHASE_ROTATE
                )
                phase_app_inv_mix = spec_utils.demud_processor(
                    org_mix, inst_source, demudder_method=DEMUD_PHASE_INVERT
                )
                phase_stem_remix = self.demix(phase_app_mix, is_demud=True)
                phase_stem_inv = self.demix(phase_app_inv_mix, is_demud=True)
                bare_stem_list = [bare_source, phase_stem_remix, phase_stem_inv]
                phase_stem = spec_utils.average_audio(
                    bare_stem_list, is_demud=True
                )
            else:
                phase_app_mix = spec_utils.demud_processor(
                    org_mix, inst_source, demudder_method=self.demudder_method
                )
                phase_stem = self.demix(phase_app_mix, is_demud=True)
            source = phase_stem if is_bare_stem else org_mix - spec_utils.match_array_shapes(phase_stem, org_mix)
            if is_bare_stem:
                source = {PRIMARY_STEM: bare_source, SECONDARY_STEM: source}
            else:
                source = {PRIMARY_STEM: source, SECONDARY_STEM: bare_source}
        return source


    def run_model(self, mix, is_match_mix=False):
        
        spek = self.stft(mix.to(self.device))*self.adjust
        spek[:, :, :3, :] *= 0 

        if is_match_mix:
            spec_pred = spek.cpu().numpy()
        else:
            spec_pred = -self.model_run(-spek)*0.5+self.model_run(spek)*0.5 if self.is_denoise else self.model_run(spek)

        return self.stft.inverse(torch.tensor(spec_pred).to(self.device)).cpu().detach().numpy()
