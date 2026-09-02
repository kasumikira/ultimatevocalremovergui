import importlib.util
import pathlib
import platform
import sys
import types
import unittest
from unittest import mock


APP_SIZE_MODULE = (
    pathlib.Path(__file__).parents[1] / "gui_data" / "app_size_values.py"
)


def load_app_sizes(screen_height, scale):
    screeninfo = types.ModuleType("screeninfo")
    screeninfo.get_monitors = lambda: [
        types.SimpleNamespace(height=screen_height, width=3840)
    ]

    dpi = types.ModuleType("gui_data.dpi")
    dpi.UI_SCALE = scale
    dpi.scale_pixels = lambda value: round(value * scale)
    dpi.scale_size = lambda size: tuple(dpi.scale_pixels(value) for value in size)

    spec = importlib.util.spec_from_file_location("app_size_under_test", APP_SIZE_MODULE)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules, {"screeninfo": screeninfo, "gui_data.dpi": dpi}
    ), mock.patch.object(platform, "system", return_value="Windows"):
        spec.loader.exec_module(module)
    return module


class AppSizeTests(unittest.TestCase):
    def test_fixed_pixel_layout_scales_at_200_percent(self):
        sizes = load_app_sizes(screen_height=2160, scale=2.0)

        self.assertEqual(sizes.WIDTH, 1360)
        self.assertEqual(sizes.COMMAND_HEIGHT, 282)
        self.assertEqual(sizes.MUSICFILE_OPEN_WIDTH, 70)
        self.assertEqual(sizes.image_scale_1, 40)

        # Font sizes are points and widget widths are characters; Tk scales
        # them independently, so they must not be multiplied here.
        self.assertEqual(sizes.FONT_SIZE_1, 8)
        self.assertEqual(sizes.MDX_CHECKBOXS_WIDTH, 14)

    def test_screen_profile_uses_logical_not_physical_height(self):
        sizes = load_app_sizes(screen_height=1440, scale=2.0)

        self.assertEqual(sizes.LOGICAL_SCREEN_HEIGHT, 720)
        self.assertEqual(sizes.COMMAND_HEIGHT, 160)


if __name__ == "__main__":
    unittest.main()
