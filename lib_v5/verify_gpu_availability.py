from gui_data.constants import *
import torch
import warnings
import gc

warnings.filterwarnings("ignore")

cuda_available = torch.cuda.is_available()

mps_available = torch.backends.mps.is_available() if is_macos else False

def detect_50_series_gpu():
    return False

is_gpu_available = cuda_available or mps_available
is_use_onnx_model = detect_50_series_gpu()

def clear_gpu_cache():
    gc.collect()
    if is_macos:
        torch.mps.empty_cache()
    else:
        torch.cuda.empty_cache()

def check_gpu_availability(is_gpu_conversion, device_set):
    device = CPU
    is_other_gpu = False

    if is_gpu_conversion >= 0:
        if mps_available:
            device, is_other_gpu = MPS_DEVICE, True
        elif cuda_available:
            device = CUDA_DEVICE if device_set == DEFAULT else f"{CUDA_DEVICE}:{device_set}"

    return (device, is_other_gpu)


detect_50_series_gpu()
