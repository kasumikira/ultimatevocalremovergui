from gui_data.constants import *
import torch
import warnings
import gc
from importlib.metadata import PackageNotFoundError

warnings.filterwarnings("ignore")

cuda_available = torch.cuda.is_available()
rocm_available = cuda_available and getattr(torch.version, "hip", None) is not None

mps_available = torch.backends.mps.is_available() if is_macos else False

is_gpu_available = cuda_available or mps_available

GPU_TYPE_CPU = "cpu"
GPU_TYPE_NVIDIA_CUDA = "nvidia_cuda"
GPU_TYPE_AMD_ROCM = "amd_rocm"
GPU_TYPE_APPLE_MPS = "apple_mps"

try:
    import onnxruntime as _onnxruntime
    onnxruntime_cuda_available = "CUDAExecutionProvider" in _onnxruntime.get_available_providers()
except (PackageNotFoundError, ImportError, OSError, RuntimeError):
    onnxruntime_cuda_available = False

def clear_gpu_cache():
    gc.collect()
    if is_macos:
        torch.mps.empty_cache()
    else:
        torch.cuda.empty_cache()

def check_gpu_availability(is_gpu_conversion, device_set):
    """Return the selected PyTorch device and its concrete backend type."""
    if is_gpu_conversion < 0:
        return CPU, GPU_TYPE_CPU

    if mps_available:
        return MPS_DEVICE, GPU_TYPE_APPLE_MPS

    if not cuda_available:
        return CPU, GPU_TYPE_CPU

    device = CUDA_DEVICE
    if device_set != DEFAULT:
        try:
            device_index = int(device_set)
        except (TypeError, ValueError):
            device_index = None

        if device_index is not None and 0 <= device_index < torch.cuda.device_count():
            device = f"{CUDA_DEVICE}:{device_index}"

    # PyTorch exposes ROCm devices through its CUDA-compatible API, so the
    # device remains ``cuda`` while the backend type distinguishes it from
    # NVIDIA CUDA.
    gpu_type = GPU_TYPE_AMD_ROCM if rocm_available else GPU_TYPE_NVIDIA_CUDA
    return device, gpu_type
