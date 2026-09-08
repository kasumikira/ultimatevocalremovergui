"""Temporary workaround for ROCm/TheRock issue #7950."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def apply_rocm_windows_offload_arch_workaround() -> None:
    """Configure the bundled ROCm runtime for compilation on Windows."""
    if os.name != "nt":
        return

    spec = importlib.util.find_spec("_rocm_sdk_core")
    if spec is None or not spec.submodule_search_locations:
        return

    runtime_root = Path(spec.submodule_search_locations[0])

    # TheRock's core wheel already ships the HIP headers Triton needs. Point
    # Triton at that SDK root instead of requiring the much larger devel wheel.
    os.environ.setdefault("ROCM_HOME", str(runtime_root))

    # Let ROCm 7.14.x's offload-arch find rocm_kpack.dll on Windows.
    runtime_bin = str(runtime_root / "bin")
    old_path = os.environ.get("PATH", "").split(os.pathsep)
    old_path = [path for path in old_path if path != runtime_bin]
    os.environ["PATH"] = os.pathsep.join((runtime_bin, *old_path))
