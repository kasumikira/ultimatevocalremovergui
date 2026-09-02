import ctypes
import importlib.util
import pathlib
import unittest
from unittest import mock


DPI_MODULE = pathlib.Path(__file__).parents[1] / "gui_data" / "dpi.py"


class FakeUser32:
    def __init__(self, dpi):
        self.dpi = dpi
        self.context = None

    def SetProcessDpiAwarenessContext(self, context):
        self.context = context
        return 1

    def GetDpiForSystem(self):
        return self.dpi


def load_dpi_module(user32):
    spec = importlib.util.spec_from_file_location("dpi_under_test", DPI_MODULE)
    module = importlib.util.module_from_spec(spec)
    with mock.patch("platform.system", return_value="Windows"), mock.patch.object(
        ctypes, "windll", mock.Mock(user32=user32), create=True
    ):
        spec.loader.exec_module(module)
    return module


class DpiTests(unittest.TestCase):
    def test_windows_system_dpi_scales_pixel_values(self):
        user32 = FakeUser32(192)
        dpi = load_dpi_module(user32)

        self.assertEqual(dpi.SYSTEM_DPI, 192)
        self.assertEqual(dpi.UI_SCALE, 2.0)
        self.assertEqual(dpi.scale_pixels(35), 70)
        self.assertEqual(dpi.scale_size((20, 30)), (40, 60))
        self.assertIsNotNone(user32.context)

        root = mock.Mock()
        with mock.patch("platform.system", return_value="Windows"):
            dpi.configure_tk_scaling(root)
        root.tk.call.assert_called_once_with("tk", "scaling", 192 / 72.0)

    def test_non_windows_uses_the_96_dpi_baseline(self):
        spec = importlib.util.spec_from_file_location("dpi_under_test", DPI_MODULE)
        dpi = importlib.util.module_from_spec(spec)
        with mock.patch("platform.system", return_value="Linux"):
            spec.loader.exec_module(dpi)

        self.assertEqual(dpi.SYSTEM_DPI, 96)
        self.assertEqual(dpi.UI_SCALE, 1.0)
        self.assertEqual(dpi.scale_pixels(35), 35)


if __name__ == "__main__":
    unittest.main()
