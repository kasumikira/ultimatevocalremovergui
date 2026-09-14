from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace

import librosa
import numpy as np
import torch

from gui_data.constants import (
    ARM,
    NON_ACCOM_STEMS,
    OPERATING_SYSTEM,
    SYSTEM_ARCH,
    SYSTEM_PROC,
    VR_ARCH_TYPE,
)
from gui_data.error_handling import ERROR_MAPPER, WINDOW_SIZE_ERROR
from lib_v5 import spec_utils
from lib_v5.vr_network import nets, nets_new
from lib_v5.vr_network.model_param_init import ModelParameters

from ..audio import banded_spectrogram, rerun_mp3
from ..contract import ProducedStems, SeparationBackend
from ..request import RunOptions
from ..state import BackendSettings, StemAlignment
from ..workflow import cached_run, create_run

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
        X_spec = banded_spectrogram(X.T, mp)

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
    def mix_to_spec(session, runtime):

        X_wave, X_spec_s = {}, {}
        
        bands_n = len(runtime.mp.param['band'])
        
        audio_file = spec_utils.write_array_to_mem(session.audio_file, subtype=session.wav_type_set)
        is_mp3 = audio_file.endswith('.mp3') if isinstance(audio_file, str) else False

        for d in range(bands_n, 0, -1):        
            bp = runtime.mp.param['band'][d]
        
            if OPERATING_SYSTEM == 'Darwin':
                wav_resolution = 'polyphase' if SYSTEM_PROC == ARM or ARM in SYSTEM_ARCH else bp['res_type']
            else:
                wav_resolution = bp['res_type']
            
            if d == bands_n: # high-end band
                X_wave[d], _ = librosa.load(audio_file, bp['sr'], False, dtype=np.float32, res_type=wav_resolution)
                X_spec_s[d] = spec_utils.wave_to_spectrogram(X_wave[d], bp['hl'], bp['n_fft'], runtime.mp, band=d, is_v51_model=runtime.is_vr_51_model)
                    
                if not np.any(X_wave[d]) and is_mp3:
                    X_wave[d] = rerun_mp3(audio_file, bp['sr'])

                if X_wave[d].ndim == 1:
                    X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
            else: # lower bands
                X_wave[d] = librosa.resample(X_wave[d+1], runtime.mp.param['band'][d+1]['sr'], bp['sr'], res_type=wav_resolution)
                X_spec_s[d] = spec_utils.wave_to_spectrogram(X_wave[d], bp['hl'], bp['n_fft'], runtime.mp, band=d, is_v51_model=runtime.is_vr_51_model)

            if d == bands_n and runtime.high_end_process != 'none':
                runtime.input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (runtime.mp.param['pre_filter_stop'] - runtime.mp.param['pre_filter_start'])
                runtime.input_high_end = X_spec_s[d][:, bp['n_fft']//2-runtime.input_high_end_h:bp['n_fft']//2, :]

        X_spec = spec_utils.combine_spectrograms(X_spec_s, runtime.mp, is_v51_model=runtime.is_vr_51_model)
        
        del X_wave, X_spec_s, audio_file

        return X_spec


    def separate(session, runtime, X_spec, device, aggressiveness):
        def _execute(X_mag_pad, roi_size):
            X_dataset = []
            patches = (X_mag_pad.shape[2] - 2 * runtime.model_run.offset) // roi_size
            total_iterations = patches//runtime.batch_size if not runtime.is_tta else (patches//runtime.batch_size)*2
            for i in range(patches):
                start = i * roi_size
                X_mag_window = X_mag_pad[:, :, start:start + runtime.window_size]
                X_dataset.append(X_mag_window)

            X_dataset = np.asarray(X_dataset)
            runtime.model_run.eval()
            with torch.no_grad():
                mask = []
                for i in range(0, patches, runtime.batch_size):
                    session.progress.completed_steps += 1
                    if session.progress.completed_steps >= total_iterations:
                        session.progress.completed_steps = total_iterations
                    session.services.set_progress_bar(
                        0.1, 0.8 / total_iterations * session.progress.completed_steps)
                    X_batch = X_dataset[i: i + runtime.batch_size]
                    X_batch = torch.from_numpy(X_batch).to(device)
                    pred = runtime.model_run.predict_mask(X_batch)
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
                if stem == session.stems.primary:
                    is_non_accom_stem = True
                    
            mask = spec_utils.adjust_aggr(mask, is_non_accom_stem, aggressiveness)

            if runtime.is_post_process:
                mask = spec_utils.merge_artifacts(mask, thres=runtime.post_process_threshold)

            y_spec = mask * X_mag * np.exp(1.j * X_phase)
            v_spec = (1 - mask) * X_mag * np.exp(1.j * X_phase)
        
            return y_spec, v_spec
        
        X_mag, X_phase = spec_utils.preprocess(X_spec)
        n_frame = X_mag.shape[2]
        pad_l, pad_r, roi_size = spec_utils.make_padding(n_frame, runtime.window_size, runtime.model_run.offset)
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_mag_pad /= X_mag_pad.max()
        mask = _execute(X_mag_pad, roi_size)
        
        if runtime.is_tta:
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


    def spec_to_wav(session, runtime, spec):
        if runtime.high_end_process.startswith('mirroring') and isinstance(runtime.input_high_end, np.ndarray) and runtime.input_high_end_h:        
            input_high_end_ = spec_utils.mirroring(runtime.high_end_process, spec, runtime.input_high_end, runtime.mp)
            wav = spec_utils.cmb_spectrogram_to_wave(spec, runtime.mp, runtime.input_high_end_h, input_high_end_, is_v51_model=runtime.is_vr_51_model)       
        else:
            wav = spec_utils.cmb_spectrogram_to_wave(spec, runtime.mp, is_v51_model=runtime.is_vr_51_model)
            
        return wav

@dataclass
class VRRuntime:
    """What one VR run owns for itself.

    The model parameters this family runs with, and the network it loads. None
    of it belongs to the shared run state: the runner carries this object and
    never reads a field of it.
    """

    mp: object
    model_samplerate: int
    model_capacity: tuple
    is_vr_51_model: bool
    high_end_process: str
    is_tta: bool
    is_post_process: bool
    batch_size: int
    window_size: int
    post_process_threshold: float
    aggressiveness: dict
    input_high_end_h: int | None = None
    input_high_end: object = None
    primary_audio: object = None
    secondary_audio: object = None
    model_run: object = None


def family_setup(model_data, request, settings: BackendSettings,
                 options: RunOptions):
    """What this family declares for one run.

    It returns the complete typed settings it aligns and the runtime state it
    owns itself.
    """
    vr_model_param = model_data.vr_model_param
    return (
        replace(settings, stems=replace(
            settings.stems, alignment=StemAlignment.ENSEMBLE_PAIR)),
        VRRuntime(
            mp=vr_model_param,
            model_samplerate=model_data.model_samplerate,
            model_capacity=model_data.model_capacity,
            is_vr_51_model=model_data.is_vr_51_model,
            high_end_process=model_data.is_high_end_process,
            is_tta=model_data.is_tta,
            is_post_process=model_data.is_post_process,
            batch_size=model_data.batch_size,
            window_size=model_data.window_size,
            post_process_threshold=model_data.post_process_threshold,
            aggressiveness={'value': model_data.aggression_setting,
                            'split_bin': vr_model_param.param['band'][1]['crop_stop'],
                            'aggr_correction': vr_model_param.param.get('aggr_correction')},
        ),
    )

@cached_run(VR_ARCH_TYPE)
def _infer(session, runtime):
    """Run this model for this file and return its output.

    Private to the family: the model's own inference result is not part of the
    interface the caller sees.
    """
    session.report_inference_start()

    device = session.device_state.torch_device

    nn_arch_sizes = [
        31191, # default
        33966, 56817, 123821, 123812, 129605, 218409, 537238, 537227]
    vr_5_1_models = [56817, 218409]
    model_size = math.ceil(os.stat(session.model_path).st_size / 1024)
    nn_arch_size = min(nn_arch_sizes, key=lambda x:abs(x-model_size))

    if nn_arch_size in vr_5_1_models or runtime.is_vr_51_model:
        runtime.model_run = nets_new.CascadedNet(runtime.mp.param['bins'] * 2,
                                              nn_arch_size,
                                              nout=runtime.model_capacity[0],
                                              nout_lstm=runtime.model_capacity[1])
        runtime.is_vr_51_model = True
    else:
        runtime.model_run = nets.determine_model_capacity(runtime.mp.param['bins'] * 2, nn_arch_size)

    runtime.model_run.load_state_dict(torch.load(session.model_path, map_location=cpu))
    runtime.model_run.to(device)

    session.report_inference_running()

    y_spec, v_spec = VRInference.separate(
        session,
        runtime,
        VRInference.mix_to_spec(session, runtime),
        device,
        runtime.aggressiveness,
    )
    session.report_inference_done()

    return y_spec, v_spec

def produce_stems(session, runtime):
    """Produce the stems the caller asked for, in the caller's order."""
    try:
        y_spec, v_spec = _infer(session, runtime)
        wanted = session.stems.select((session.stems.primary, session.stems.secondary))
        stems = {}
        for stem_name in wanted:
            if stem_name == session.stems.primary:
                stems[stem_name] = primary_stem_audio(session, runtime, y_spec)
            elif stem_name == session.stems.secondary:
                stems[stem_name] = secondary_stem_audio(session, runtime, v_spec)
        return ProducedStems(stems=stems)
    finally:
        runtime.model_run = None

def primary_stem_audio(session, runtime, y_spec):
    if not isinstance(runtime.primary_audio, np.ndarray):
        runtime.primary_audio = VRInference.spec_to_wav(session, runtime, y_spec).T
        if runtime.model_samplerate != 44100:
            runtime.primary_audio = librosa.resample(runtime.primary_audio.T, orig_sr=runtime.model_samplerate, target_sr=44100).T
    return runtime.primary_audio

def secondary_stem_audio(session, runtime, v_spec):
    if not isinstance(runtime.secondary_audio, np.ndarray):
        runtime.secondary_audio = VRInference.spec_to_wav(session, runtime, v_spec).T
        if not runtime.model_samplerate == 44100:
            runtime.secondary_audio = librosa.resample(runtime.secondary_audio.T, orig_sr=runtime.model_samplerate, target_sr=44100).T
    return runtime.secondary_audio


def create_backend(request, services, options=None):
    """This family wired for one run: its state, plus the functions the
    shared runner calls.

    Nothing is loaded or computed here: the family's model runs when the
    runner calls ``produce_stems``.
    """
    session, runtime = create_run(request, services, family_setup, options)
    return SeparationBackend(
        session=session,
        runtime=runtime,
        produce_stems=produce_stems,
    )
