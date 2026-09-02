"""Windows DPI helpers used before the Tk UI is created."""

import ctypes
import platform


BASE_DPI = 96
_SYSTEM_AWARE = -2


def enable_dpi_awareness():
    """Make Windows render the application at the system DPI.

    UVR uses a fixed-pixel layout and currently does not relayout itself when a
    window is moved between monitors. System awareness therefore provides a
    sharp UI on the system-DPI monitor while Windows can still scale the window
    when it is moved to a monitor with a different DPI.
    """
    if platform.system() != "Windows":
        return

    try:
        user32 = ctypes.windll.user32
        set_context = getattr(user32, "SetProcessDpiAwarenessContext", None)
        if set_context is not None:
            set_context(ctypes.c_void_p(_SYSTEM_AWARE))
        else:
            user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        # Keep compatibility with older Windows versions and non-standard
        # Python runtimes. Windows will fall back to its legacy DPI behavior.
        pass


def get_system_dpi():
    """Return the Windows system DPI, falling back to the 96-DPI baseline."""
    if platform.system() != "Windows":
        return BASE_DPI

    try:
        get_dpi = getattr(ctypes.windll.user32, "GetDpiForSystem", None)
        if get_dpi is not None:
            dpi = int(get_dpi())
            if dpi > 0:
                return dpi
    except (AttributeError, OSError, TypeError, ValueError):
        pass

    return BASE_DPI


enable_dpi_awareness()
SYSTEM_DPI = get_system_dpi()
UI_SCALE = SYSTEM_DPI / BASE_DPI


def scale_pixels(value):
    """Scale a coordinate or size expressed in 96-DPI pixels."""
    return round(value * UI_SCALE)


def scale_size(size):
    """Scale a two-dimensional pixel size."""
    return tuple(scale_pixels(value) for value in size)


def configure_tk_scaling(root):
    """Set Tk's pixels-per-point value before fonts and widgets are created."""
    if platform.system() == "Windows":
        root.tk.call("tk", "scaling", SYSTEM_DPI / 72.0)
