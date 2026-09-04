from gui_data.constants import *
import torch
import warnings
import gc

warnings.filterwarnings("ignore")

cuda_available = torch.cuda.is_available()
is_windows_ = OPERATING_SYSTEM == "Windows"
is_import_direct_ml = not cuda_available and is_windows_

if is_import_direct_ml:
    import torch_directml

mps_available = torch.backends.mps.is_available() if is_macos else False

def detect_50_series_gpu():
    return False

def get_gpu_info():
    directml_device, directml_available = DIRECTML_DEVICE, False

    if is_import_direct_ml:
        directml_available = torch_directml.is_available()

        if directml_available:
            directml_device = str(torch_directml.device()).partition(":")[0]

    return (directml_device, directml_available)

DIRECTML_DEVICE, directml_available = get_gpu_info()
is_choose_arch = cuda_available and directml_available
is_directml_only = not cuda_available and directml_available
is_cuda_only = cuda_available and not directml_available
is_gpu_available = cuda_available or directml_available or mps_available
is_use_onnx_model = detect_50_series_gpu()

def clear_gpu_cache():
    gc.collect()
    if is_macos:
        torch.mps.empty_cache()
    else:
        torch.cuda.empty_cache()

def check_gpu_availability(is_gpu_conversion, device_set, is_use_directml):
    device = CPU
    is_other_gpu = False
    # is_using_directml = False

    if is_gpu_conversion >= 0:
        if mps_available:
            device, is_other_gpu = MPS_DEVICE, True
        else:
            device_prefix = None
            if device_set != DEFAULT:
                device_prefix = DIRECTML_DEVICE if is_use_directml and directml_available else CUDA_DEVICE

            if directml_available and is_use_directml:
                device = torch_directml.device() if not device_prefix else f"{device_prefix}:{device_set}"
                is_other_gpu = True

            elif cuda_available and not is_use_directml:
                device = CUDA_DEVICE if not device_prefix else f"{device_prefix}:{device_set}"

    return (device, is_other_gpu)


detect_50_series_gpu()
