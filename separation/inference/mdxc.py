from __future__ import annotations

import torch
from ml_collections import ConfigDict
from torch import nn

from lib_v5.bandit.core.model import MultiMaskMultiSourceBandSplitRNNSimple
from lib_v5.bandit_v2.bandit import Bandit
from lib_v5.roformer.bs_roformer import BSRoformer
from lib_v5.roformer.bs_roformer_new import BSRoformer as BSRoformerNew
from lib_v5.roformer.mel_band_roformer import MelBandRoformer
from lib_v5.roformer.mel_band_roformer_new import MelBandRoformer as MelBandRoformerNew
from lib_v5.scnet.scnet import SCNet
from lib_v5.tfc_tdf_v3 import TFC_TDF_net
from lib_v5.verify_gpu_availability import GPU_TYPE_APPLE_MPS

MDXC_MODELS = {
    'BS-Roformer': ('BSRoformer', 'model'),
    'BS-Roformer v2': ('BSRoformerNew', 'model'),
    'BS-Roformer New': ('BSRoformerNew', 'model'),
    'MelBand-Roformer': ('MelBandRoformer', 'model'),
    'MelBand-Roformer v2': ('MelBandRoformerNew', 'model'),
    'MelBand-Roformer New': ('MelBandRoformerNew', 'model'),
    'SCNet': ('SCNet', 'model'),
    'Bandit': ('MultiMaskMultiSourceBandSplitRNNSimple', 'model'),
    'Bandit v2': ('Bandit', 'kwargs'),
    'Bandit 2': ('Bandit', 'kwargs'),
}


def build_mdxc_model(model_type, config, device, is_roformer=False, constructors=None):
    """Construct a network without loading weights or running inference.

    Unknown legacy types retain the original MDX23C/config-based fallback.
    The optional namespace lets callers retain their existing constructor bindings.
    """
    if constructors is None:
        constructors = globals()
    entry = MDXC_MODELS.get(model_type)
    if entry is not None:
        symbol, section = entry
        return constructors[symbol](**getattr(config, section))
    if is_roformer:
        if 'num_bands' in config.model:
            return constructors['MelBandRoformer'](**config.model)
        if 'freqs_per_bands' in config.model:
            return constructors['BSRoformer'](**config.model)
        raise ValueError('Unknown model type in the configuration.')
    return constructors['TFC_TDF_net'](config, device=device)


class MDXCInference:
    def _getWindowingArray(self, window_size: int, fade_size: int, device) -> torch.Tensor:
        fadein = torch.linspace(0, 1, fade_size).to(device)
        fadeout = torch.linspace(1, 0, fade_size).to(device)
        window = torch.ones(window_size).to(device)
        window[-fade_size:] = fadeout
        window[:fade_size] = fadein
        return window


    def overlap_add(self, result, x, l, j, start, window):
        if self.gpu_type == GPU_TYPE_APPLE_MPS:
            x = x.to(self.device)
        result[..., start:start + l] += x[j, ..., :l] * window[..., :l]
        return result


    def find_hop_size(self, config_):
        def _find_hop(config):
            if isinstance(config, dict) or isinstance(config, ConfigDict):
                for key, value in config.items():
                    if key in ('hop_size', 'hop_length'):
                        return value
                    elif isinstance(value, (dict, ConfigDict)):
                        result = _find_hop(value)
                        if result is not None:
                            return result
            elif hasattr(config, '__dict__'):
                return _find_hop(config.__dict__)
        set_chunk = config_.audio.chunk_size
        dim_t_c = getattr(config_.inference, 'dim_t', 256) - 1
        hop_size = _find_hop(config_)
        if hop_size and hop_size * dim_t_c == set_chunk:
            return hop_size
        else:
            return set_chunk // dim_t_c


    def _load_model(self):
        """Keep checkpoint interpretation and strict loading compatible."""
        model = build_mdxc_model(
            self.mdx_model_type, self.gen_model_config, self.device,
            self.is_roformer,
        )
        checkpoint = torch.load(self.model_path, map_location='cpu')
        model = model if not isinstance(model, torch.nn.DataParallel) else model.module
        model.load_state_dict(checkpoint)
        model.to(self.device).eval()
        return model


    def _predict_chunks(self, model, mix, chunk_size, num_overlap, num_instruments, chunk_add):
        """Run the legacy overlap-add algorithm; no loading or stem processing."""
        device = self.device
        step = int(chunk_size // num_overlap)
        fade_size = chunk_size // 10
        border = chunk_size - step
        batch_size = 1
        length_init = mix.shape[-1]
        windowing_array = self._getWindowingArray(chunk_size, fade_size, device)
        if length_init > 2 * border and border:
            mix = nn.functional.pad(mix, (border, border), mode='reflect')
        batch_len = int(mix.shape[1] / step)
        if self.is_demud:
            batch_len = batch_len * chunk_add
        with torch.inference_mode() if self.is_use_torch_inference_mode else torch.no_grad():
            req_shape = (num_instruments,) + mix.shape
            result = torch.zeros(req_shape, dtype=torch.float32, device=device)
            counter = torch.zeros(req_shape, dtype=torch.float32, device=device)
            batch_data = []
            batch_locations = []
            i = 0
            while i < mix.shape[1]:
                part = mix[:, i:i + chunk_size].to(device)
                length = part.shape[-1]
                if length > chunk_size // 2:
                    pad_mode = 'reflect'
                else:
                    pad_mode = 'constant'
                part = nn.functional.pad(part, (0, chunk_size - length), mode=pad_mode, value=0)
                batch_data.append(part)
                batch_locations.append((i, length))
                i += step
                # Process in batches
                if len(batch_data) >= batch_size or (i >= mix.shape[1]):
                    arr = torch.stack(batch_data, dim=0)
                    x = model(arr)
                    window = windowing_array.clone()
                    if i - step == 0:
                        window[:fade_size] = 1
                    elif i >= mix.shape[1]:
                        window[-fade_size:] = 1
                    for j, (start, seg_len) in enumerate(batch_locations):
                        self.running_inference_progress_bar(batch_len)
                        min_length = min(result[..., start:start + seg_len].shape[-1], x[j, ..., :seg_len].shape[-1], window[..., :seg_len].shape[-1])
                        result = self.overlap_add(result, x, min_length, j, start, window)
                        counter[..., start:start + seg_len] += window[..., :seg_len]
                    batch_data = []
                    batch_locations = []
            # Normalize by the overlap counter and remove padding
            estimated_sources = result / counter.clamp(min=1e-10)
            if length_init > 2 * (chunk_size - step) and chunk_size - step > 0:
                estimated_sources = estimated_sources[..., chunk_size - step:-(chunk_size - step)]
            estimated_sources = estimated_sources.cpu().numpy()
        return estimated_sources
