from __future__ import annotations

import numpy as np
import torch

from demucs.apply import apply_model
from demucs.utils import apply_model_v1, apply_model_v2
from gui_data.constants import DEMUCS_V1, DEMUCS_V2
from lib_v5 import spec_utils


class DemucsInference:
    def demix_demucs(self, mix):
        
        org_mix = mix
        
        if self.is_pitch_change:
            mix, sr_pitched = spec_utils.change_pitch_semitones(mix, 44100, semitone_shift=-self.semitone_shift)
        
        processed = {}
        mix = torch.tensor(mix, dtype=torch.float32)
        ref = mix.mean(0)        
        mix = (mix - ref.mean()) / ref.std()
        mix_infer = mix 
        
        with torch.no_grad():
            if self.demucs_version == DEMUCS_V1:
                sources = apply_model_v1(self.demucs, 
                                            mix_infer.to(self.device), 
                                            self.shifts, 
                                            self.is_split_mode,
                                            set_progress_bar=self.set_progress_bar)
            elif self.demucs_version == DEMUCS_V2:
                sources = apply_model_v2(self.demucs, 
                                            mix_infer.to(self.device), 
                                            self.shifts,
                                            self.is_split_mode,
                                            self.overlap,
                                            set_progress_bar=self.set_progress_bar)
            else:
                print('self.shifts', self.shifts)
                print('is_split_mode', self.is_split_mode)
                print('overlap', self.overlap)
                sources = apply_model(self.demucs, 
                                        mix_infer[None], 
                                        self.shifts,
                                        self.is_split_mode,
                                        self.overlap,
                                        static_shifts=1 if self.shifts == 0 else self.shifts,
                                        set_progress_bar=self.set_progress_bar,
                                        device=self.device)[0]
        
        sources = (sources * ref.std() + ref.mean()).cpu().numpy()
        sources[[0,1]] = sources[[1,0]]
        processed[mix] = sources[:,:,0:None].copy()
        sources = list(processed.values())
        sources = [s[:,:,0:None] for s in sources]
        #sources = [self.pitch_fix(s[:,:,0:None], sr_pitched, org_mix) if self.is_pitch_change else s[:,:,0:None] for s in sources]
        sources = np.concatenate(sources, axis=-1)
                     
        if self.is_pitch_change:
            sources = np.stack([self.pitch_fix(stem, sr_pitched, org_mix) for stem in sources])
                        
        return sources
