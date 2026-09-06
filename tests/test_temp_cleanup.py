import ast
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


def load_delete_temps(namespace):
    # Extract the real method without importing UVR's GUI and model runtimes.
    source = Path(__file__).parents[1] / 'UVR.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    window = next(node for node in tree.body
                  if isinstance(node, ast.ClassDef) and node.name == 'MainWindow')
    method = next(node for node in window.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'delete_temps')
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(module, str(source), 'exec'), namespace)
    return namespace['delete_temps']


class TempCleanupTests(unittest.TestCase):
    def test_cleanup_preserves_documents_and_removes_download_temps(self):
        for is_start_up in (True, False):
            with self.subTest(is_start_up=is_start_up), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                names = ('BASE_PATH', 'VR_MODELS_DIR', 'MDX_MODELS_DIR',
                         'DEMUCS_MODELS_DIR', 'DEMUCS_NEWER_REPO_DIR')
                directories = {name: root / name for name in names}
                for directory in directories.values():
                    directory.mkdir()
                    for name in ('requirements.txt', 'notes.txt', 'LICENSE.txt',
                                 'write_error.txt', 'notes.TXT', 'private.aes', 'model.pth',
                                 'unfinished.pth.tmp'):
                        (directory / name).write_text('keep contents', encoding='utf-8')
                    (directory / 'folder.tmp').mkdir()
                    (directory / 'folder.tmp' / 'notes.txt').write_text('nested')

                splash = root / 'splash.txt'
                splash.write_text('1')
                patch = root / 'test_patch.exe'
                patch.write_bytes(b'patch')
                namespace = dict(directories, os=os, SPLASH_DOC=str(splash),
                                 current_patch=str(root / 'test_patch'),
                                 application_extension='.exe',
                                 TEMP_FILE_DELETION_TEXT='Temp File Deletion',
                                 error_text=lambda title, error: str(error))
                cleanup = load_delete_temps(namespace)
                window = mock.Mock()
                # Repeat to cover subsequent download/exit cleanup calls too.
                for _ in range(2):
                    cleanup(window, is_start_up=is_start_up)
                window.error_log_var.set.assert_not_called()
                self.assertFalse(patch.exists())
                self.assertEqual(splash.exists(), is_start_up)
                for directory in directories.values():
                    self.assertFalse((directory / 'unfinished.pth.tmp').exists())
                    for name in ('requirements.txt', 'notes.txt', 'LICENSE.txt',
                                 'write_error.txt', 'notes.TXT', 'private.aes', 'model.pth'):
                        self.assertEqual((directory / name).read_text(), 'keep contents')
                    self.assertEqual((directory / 'folder.tmp' / 'notes.txt').read_text(),
                                     'nested')


if __name__ == '__main__':
    unittest.main()
