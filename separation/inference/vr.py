from __future__ import annotations

import os

import librosa
import numpy as np
import torch

from gui_data.constants import (
    ARM,
    NON_ACCOM_STEMS,
    OPERATING_SYSTEM,
    SYSTEM_ARCH,
    SYSTEM_PROC,
)
from gui_data.error_handling import ERROR_MAPPER, WINDOW_SIZE_ERROR
from lib_v5 import spec_utils
from lib_v5.vr_network import nets_new
from lib_v5.vr_network.model_param_init import ModelParameters

from ..audio import loading_mix, rerun_mp3

cpu = torch.device('cpu')


def vr_denoiser(X, device, hop_length=1024, n_fft=2048, cropsize=256, is_deverber=False, model_path=None):
    batchsize = 4

    if is_deverber:
        nout, nout_lstm = 64, 128
        mp = ModelParameters(os.path.join('lib_v5', 'vr_network', 'modelparams', '4band_v3.json'))
        n_fft = mp.param['bins'] * 2
    else:
        mp = None
        hop_length=1024
        nout, nout_lstm = 16, 128

    model = nets_new.CascadedNet(n_fft, nout=nout, nout_lstm=nout_lstm)
    model.load_state_dict(torch.load(model_path, map_location=cpu))
    model.to(device)

    if mp is None:
        X_spec = spec_utils.wave_to_spectrogram_old(X, hop_length, n_fft)
    else:
        X_spec = loading_mix(X.T, mp)

    #PreProcess
    X_mag = np.abs(X_spec)
    X_phase = np.angle(X_spec)

    #Sep
    n_frame = X_mag.shape[2]
    pad_l, pad_r, roi_size = spec_utils.make_padding(n_frame, cropsize, model.offset)
    X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
    X_mag_pad /= X_mag_pad.max()

    X_dataset = []
    patches = (X_mag_pad.shape[2] - 2 * model.offset) // roi_size
    for i in range(patches):
        start = i * roi_size
        X_mag_crop = X_mag_pad[:, :, start:start + cropsize]
        X_dataset.append(X_mag_crop)

    X_dataset = np.asarray(X_dataset)

    model.eval()

    with torch.no_grad():
        mask = []
        # To reduce the overhead, dataloader is not used.
        for i in range(0, patches, batchsize):
            X_batch = X_dataset[i: i + batchsize]
            X_batch = torch.from_numpy(X_batch).to(device)

            pred = model.predict_mask(X_batch)

            pred = pred.detach().cpu().numpy()
            pred = np.concatenate(pred, axis=2)
            mask.append(pred)

        mask = np.concatenate(mask, axis=2)

    mask = mask[:, :, :n_frame]

    #Post Proc
    if is_deverber:
        v_spec = mask * X_mag * np.exp(1.j * X_phase)
        y_spec = (1 - mask) * X_mag * np.exp(1.j * X_phase)
    else:
        v_spec = (1 - mask) * X_mag * np.exp(1.j * X_phase)

    if mp is None:
        wave = spec_utils.spectrogram_to_wave_old(v_spec, hop_length=1024)
    else:
        wave = spec_utils.cmb_spectrogram_to_wave(v_spec, mp, is_v51_model=True).T

    wave = spec_utils.match_array_shapes(wave, X)

    if is_deverber:
        wave_2 = spec_utils.cmb_spectrogram_to_wave(y_spec, mp, is_v51_model=True).T
        wave_2 = spec_utils.match_array_shapes(wave_2, X)
        return wave, wave_2
    else:
        return wave


class VRInference:
    def loading_mix(self):

        X_wave, X_spec_s = {}, {}
        
        bands_n = len(self.mp.param['band'])
        
        audio_file = spec_utils.write_array_to_mem(self.audio_file, subtype=self.wav_type_set)
        is_mp3 = audio_file.endswith('.mp3') if isinstance(audio_file, str) else False

        for d in range(bands_n, 0, -1):        
            bp = self.mp.param['band'][d]
        
            if OPERATING_SYSTEM == 'Darwin':
                wav_resolution = 'polyphase' if SYSTEM_PROC == ARM or ARM in SYSTEM_ARCH else bp['res_type']
            else:
                wav_resolution = bp['res_type']
        
            if d == bands_n: # high-end band
                X_wave[d], _ = librosa.load(audio_file, bp['sr'], False, dtype=np.float32, res_type=wav_resolution)
                X_spec_s[d] = spec_utils.wave_to_spectrogram(X_wave[d], bp['hl'], bp['n_fft'], self.mp, band=d, is_v51_model=self.is_vr_51_model)
                    
                if not np.any(X_wave[d]) and is_mp3:
                    X_wave[d] = rerun_mp3(audio_file, bp['sr'])

                if X_wave[d].ndim == 1:
                    X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
            else: # lower bands
                X_wave[d] = librosa.resample(X_wave[d+1], self.mp.param['band'][d+1]['sr'], bp['sr'], res_type=wav_resolution)
                X_spec_s[d] = spec_utils.wave_to_spectrogram(X_wave[d], bp['hl'], bp['n_fft'], self.mp, band=d, is_v51_model=self.is_vr_51_model)

            if d == bands_n and self.high_end_process != 'none':
                self.input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (self.mp.param['pre_filter_stop'] - self.mp.param['pre_filter_start'])
                self.input_high_end = X_spec_s[d][:, bp['n_fft']//2-self.input_high_end_h:bp['n_fft']//2, :]

        X_spec = spec_utils.combine_spectrograms(X_spec_s, self.mp, is_v51_model=self.is_vr_51_model)
        
        del X_wave, X_spec_s, audio_file

        return X_spec


    def inference_vr(self, X_spec, device, aggressiveness):
        def _execute(X_mag_pad, roi_size):
            X_dataset = []
            patches = (X_mag_pad.shape[2] - 2 * self.model_run.offset) // roi_size
            total_iterations = patches//self.batch_size if not self.is_tta else (patches//self.batch_size)*2
            for i in range(patches):
                start = i * roi_size
                X_mag_window = X_mag_pad[:, :, start:start + self.window_size]
                X_dataset.append(X_mag_window)

            X_dataset = np.asarray(X_dataset)
            self.model_run.eval()
            with torch.no_grad():
                mask = []
                for i in range(0, patches, self.batch_size):
                    self.progress_value += 1
                    if self.progress_value >= total_iterations:
                        self.progress_value = total_iterations
                    self.set_progress_bar(0.1, 0.8/total_iterations*self.progress_value)
                    X_batch = X_dataset[i: i + self.batch_size]
                    X_batch = torch.from_numpy(X_batch).to(device)
                    pred = self.model_run.predict_mask(X_batch)
                    if not pred.size()[3] > 0:
                        raise Exception(ERROR_MAPPER[WINDOW_SIZE_ERROR])
                    pred = pred.detach().cpu().numpy()
                    pred = np.concatenate(pred, axis=2)
                    mask.append(pred)
                if len(mask) == 0:
                    raise Exception(ERROR_MAPPER[WINDOW_SIZE_ERROR])
                
                mask = np.concatenate(mask, axis=2)
            return mask

        def postprocess(mask, X_mag, X_phase):
            is_non_accom_stem = False
            for stem in NON_ACCOM_STEMS:
                if stem == self.primary_stem:
                    is_non_accom_stem = True
                    
            mask = spec_utils.adjust_aggr(mask, is_non_accom_stem, aggressiveness)

            if self.is_post_process:
                mask = spec_utils.merge_artifacts(mask, thres=self.post_process_threshold)

            y_spec = mask * X_mag * np.exp(1.j * X_phase)
            v_spec = (1 - mask) * X_mag * np.exp(1.j * X_phase)
        
            return y_spec, v_spec
        
        X_mag, X_phase = spec_utils.preprocess(X_spec)
        n_frame = X_mag.shape[2]
        pad_l, pad_r, roi_size = spec_utils.make_padding(n_frame, self.window_size, self.model_run.offset)
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_mag_pad /= X_mag_pad.max()
        mask = _execute(X_mag_pad, roi_size)
        
        if self.is_tta:
            pad_l += roi_size // 2
            pad_r += roi_size // 2
            X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
            X_mag_pad /= X_mag_pad.max()
            mask_tta = _execute(X_mag_pad, roi_size)
            mask_tta = mask_tta[:, :, roi_size // 2:]
            mask = (mask[:, :, :n_frame] + mask_tta[:, :, :n_frame]) * 0.5
        else:
            mask = mask[:, :, :n_frame]

        y_spec, v_spec = postprocess(mask, X_mag, X_phase)
        
        return y_spec, v_spec


    def spec_to_wav(self, spec):
        if self.high_end_process.startswith('mirroring') and isinstance(self.input_high_end, np.ndarray) and self.input_high_end_h:        
            input_high_end_ = spec_utils.mirroring(self.high_end_process, spec, self.input_high_end, self.mp)
            wav = spec_utils.cmb_spectrogram_to_wave(spec, self.mp, self.input_high_end_h, input_high_end_, is_v51_model=self.is_vr_51_model)       
        else:
            wav = spec_utils.cmb_spectrogram_to_wave(spec, self.mp, is_v51_model=self.is_vr_51_model)
            
        return wav
