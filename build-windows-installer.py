from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import PyInstaller.__main__
from rocm_sdk import _dist_info

from __version__ import VERSION

ROOT = Path(__file__).resolve().parent
BUILDROOT = ROOT / "build"
PYINSTALLER = BUILDROOT / "pyinstaller"
DIST = PYINSTALLER / "dist"
WORK = PYINSTALLER / "work"
VENDOR = BUILDROOT / "vendor"

APP_NAME = "UVR"
ICON = ROOT / "gui_data" / "img" / "GUI-Icon.ico"


def assert_project_environment() -> None:
    expected = (ROOT / ".venv").resolve()
    actual = Path(sys.prefix).resolve()

    if actual != expected:
        raise SystemExit(
            "Run this build through the uv project environment:\n"
            "    uv sync\n"
            "    uv run python packaging/build.py\n"
            f"Expected sys.prefix: {expected}\n"
            f"Actual sys.prefix:   {actual}"
        )

    if sys.version_info[:2] != (3, 10) or sys.maxsize <= 2**32:
        raise SystemExit("The build environment must be 64-bit CPython 3.10.")


def data_arg(source: Path, destination: str) -> str:
    # PyInstaller accepts os.pathsep as the source/destination separator.
    return f"{source}{os.pathsep}{destination}"


def prepare_vendor_binaries() -> None:
    VENDOR.mkdir(parents=True, exist_ok=True)

    downloads = (
        (
            "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
            {"ffmpeg.exe": "/bin/ffmpeg.exe"},
        ),
        (
            "https://breakfastquay.com/files/releases/rubberband-3.1.2-gpl-executable-windows.zip",
            {
                "rubberband.exe": "/rubberband.exe",
                "sndfile.dll": "/sndfile.dll",
            },
        ),
    )

    for url, files in downloads:
        if all((VENDOR / name).is_file() for name in files):
            continue

        archive = VENDOR / "download.zip"
        urllib.request.urlretrieve(url, archive)

        with zipfile.ZipFile(archive) as z:
            for output, suffix in files.items():
                member = next(name for name in z.namelist() if name.endswith(suffix))
                (VENDOR / output).write_bytes(z.read(member))

        archive.unlink()


def add_runtime_resources(args: list[str]) -> None:
    # Add the GUI data, models, and lib_v5 directories to the PyInstaller build.
    for path in ("gui_data", "models", "lib_v5"):
        args.extend(["--add-data", data_arg(ROOT / path, path)])
    args.extend(["--collect-submodules", "demucs"])
    args.extend(["--collect-binaries", "samplerate"])

    # target_families = (
    #     _dist_info.WINDOWS_TARGET_FAMILIES
    #     or _dist_info.AVAILABLE_TARGET_FAMILIES
    # )
    # target_families = sorted({ family.split(":")[0] for family in target_families })
    core_modules = _dist_info.ALL_PACKAGES["core"].get_py_package_name()
    libraries_module = _dist_info.ALL_PACKAGES["libraries"].get_py_package_name()

    for module_name in (core_modules, libraries_module):
        if importlib.util.find_spec(module_name) is None:
            raise SystemExit(
                f"ROCm runtime package is not importable: {module_name}"
            )
        args.extend([
            "--hidden-import", module_name,
            "--collect-data", module_name,
            "--collect-binaries", module_name,
        ])


def add_vendor_binaries(args: list[str]) -> None:
    for name in ("ffmpeg.exe", "rubberband.exe", "sndfile.dll"):
        source = VENDOR / name
        if not source.is_file():
            raise SystemExit(f"Missing required vendor binary: {source}")
        args.extend(["--add-binary", data_arg(source, ".")])


def run_pyinstaller(clean: bool, debug_console: bool = False) -> None:
    WORK.mkdir(parents=True, exist_ok=True)

    args = [
        str(ROOT / "UVR.py"),
        "--name", APP_NAME,
        "--onedir",
        "--contents-directory", ".",
        "--noconfirm",
        "--noupx",
        "--icon", str(ICON),
        "--distpath", str(DIST),
        "--workpath", str(WORK),
        "--specpath", str(WORK),
    ]

    if debug_console:
        args.extend(["--console", "--debug", "all"])
    else:
        args.append("--windowed")

    if clean:
        args.append("--clean")

    add_runtime_resources(args)
    add_vendor_binaries(args)

    PyInstaller.__main__.run(args)

    exe = DIST / APP_NAME / f"{APP_NAME}.exe"
    if not exe.is_file():
        raise SystemExit(f"PyInstaller completed without producing {exe}")

    print(f"Built application: {exe}")


def read_uvr_version() -> str:
    return VERSION.split('v')[-1]


def run_inno_setup() -> None:
    # Deliberately target one specific Windows build environment.
    program_files_x86 = os.environ["ProgramFiles(x86)"]
    iscc = Path(program_files_x86) / "Inno Setup 6" / "ISCC.exe"
    if not iscc.is_file():
        raise SystemExit(f"Inno Setup 6 not found: {iscc}")

    installer = ROOT / "installer.iss"
    if not installer.is_file():
        raise SystemExit(f"Missing installer definition: {installer}")

    version = read_uvr_version()
    subprocess.run(
        [str(iscc), f"/DAppVersion={version}", str(installer)],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--no-installer", action="store_true")
    parser.add_argument("--debug-console", action="store_true")
    args = parser.parse_args()

    assert_project_environment()
    prepare_vendor_binaries()
    run_pyinstaller(clean=not args.no_clean, debug_console=args.debug_console)

    if not args.no_installer:
        run_inno_setup()


if __name__ == "__main__":
    main()
