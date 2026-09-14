"""Audio separation workflows and their public entry points.

A caller describes a run with :class:`RunRequest` and the application side of it
with :class:`RunServices`, starts it with ``run_separator``, and reads the
outcome off the :class:`SeparationResult` it returns. Everything else — the
runner, the backends, the shared run state, the sub-run launchers — is
implementation, and lives in the module that owns it.
"""

from separation.audio import save_format
from separation.contract import SeparationResult
from separation.factory import run_separator
from separation.request import RunRequest, RunServices, StemSelection

__all__ = [
    'RunRequest',
    'RunServices',
    'SeparationResult',
    'StemSelection',
    'run_separator',
    'save_format',
]
