from __future__ import annotations

import os

import audioread
import librosa
import numpy as np
import pydub
from scipy import signal

from gui_data.constants import (
    ARM,
    FLAC,
    MP3,
    OPERATING_SYSTEM,
    SYSTEM_ARCH,
    SYSTEM_PROC,
    WAV,
)
from lib_v5 import spec_utils


def prepare_mix(mix):
    
    audio_path = mix

    if not isinstance(mix, np.ndarray):
        mix, sr = librosa.load(mix, mono=False, sr=44100)
    else:
        mix = mix.T

    if isinstance(audio_path, str):
        if not np.any(mix) and audio_path.endswith('.mp3'):
            mix = rerun_mp3(audio_path)

    if mix.ndim == 1:
        mix = np.asfortranarray([mix,mix])

    return mix


def rerun_mp3(audio_file, sample_rate=44100):

    with audioread.audio_open(audio_file) as f:
        track_length = int(f.duration)

    return librosa.load(audio_file, duration=track_length, mono=False, sr=sample_rate)[0]


def save_format(audio_path, save_format, mp3_bit_set):
    
    if not save_format == WAV:
        
        if OPERATING_SYSTEM == 'Darwin':
            FFMPEG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ffmpeg')
            pydub.AudioSegment.converter = FFMPEG_PATH
        
        musfile = pydub.AudioSegment.from_wav(audio_path)
        
        if save_format == FLAC:
            audio_path_flac = audio_path.replace(".wav", ".flac")
            musfile.export(audio_path_flac, format="flac")  
        
        if save_format == MP3:
            audio_path_mp3 = audio_path.replace(".wav", ".mp3")
            try:
                musfile.export(audio_path_mp3, format="mp3", bitrate=mp3_bit_set, codec="libmp3lame")
            except Exception as e:
                print(e)
                musfile.export(audio_path_mp3, format="mp3", bitrate=mp3_bit_set)
        
        try:
            os.remove(audio_path)
        except Exception as e:
            print(e)


def pitch_shift(mix):
    new_sr = 31183

    # Resample audio file
    resampled_audio = signal.resample_poly(mix, new_sr, 44100)
    
    return resampled_audio


def list_to_dictionary(lst):
    dictionary = {item: index for index, item in enumerate(lst)}
    return dictionary


def loading_mix(X, mp):

    X_wave, X_spec_s = {}, {}
    
    bands_n = len(mp.param['band'])
    
    for d in range(bands_n, 0, -1):        
        bp = mp.param['band'][d]
    
        if OPERATING_SYSTEM == 'Darwin':
            wav_resolution = 'polyphase' if SYSTEM_PROC == ARM or ARM in SYSTEM_ARCH else bp['res_type']
        else:
            wav_resolution = 'polyphase'#bp['res_type']
    
        if d == bands_n: # high-end band
            X_wave[d] = X

        else: # lower bands
            X_wave[d] = librosa.resample(X_wave[d+1], mp.param['band'][d+1]['sr'], bp['sr'], res_type=wav_resolution)
            
        X_spec_s[d] = spec_utils.wave_to_spectrogram(X_wave[d], bp['hl'], bp['n_fft'], mp, band=d, is_v51_model=True)
        
        # if d == bands_n and is_high_end_process:
        #     input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (mp.param['pre_filter_stop'] - mp.param['pre_filter_start'])
        #     input_high_end = X_spec_s[d][:, bp['n_fft']//2-input_high_end_h:bp['n_fft']//2, :]

    X_spec = spec_utils.combine_spectrograms(X_spec_s, mp)
    
    del X_wave, X_spec_s

    return X_spec
