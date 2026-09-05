"""Temporary workaround for ROCm/TheRock issue #7950."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def apply_rocm_windows_offload_arch_workaround() -> None:
    """Let ROCm 7.14.x's offload-arch find rocm_kpack.dll on Windows."""
    if os.name != "nt":
        return

    spec = importlib.util.find_spec("_rocm_sdk_core")
    if spec is None or not spec.submodule_search_locations:
        return

    runtime_bin = str(Path(spec.submodule_search_locations[0]) / "bin")
    old_path = os.environ.get("PATH", "").split(os.pathsep)
    old_path = [path for path in old_path if path != runtime_bin]
    os.environ["PATH"] = os.pathsep.join((runtime_bin, *old_path))
