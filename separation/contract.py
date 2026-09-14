"""Typed boundaries between inference, orchestration and output.

Every stem is a 44100 Hz numpy array shaped (samples, 2). A backend returns
exactly the requested names; display labels and filename suffixes are output
concerns. Cached inference payloads are private to each backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

import numpy as np

from separation.request import RunKind, RunOptions

if TYPE_CHECKING:
    from separation.workflow import SeparationSession

Runtime = TypeVar('Runtime')
StemMap = dict[str, np.ndarray]


@dataclass(frozen=True)
class SeparationResult:
    """Named audio returned for every run, whether or not files were saved."""

    stems: StemMap = field(default_factory=dict)


@dataclass(frozen=True)
class ProducedStems:
    """Requested audio, plus optional export-only derivatives.

    Only ``stems`` participates in secondary blending, the vocal chain and the
    returned result. ``extra_stems`` holds separately named derivatives such as
    a Demucs preprocessed instrumental, with no additional model dispatch.
    """

    stems: StemMap = field(default_factory=dict)
    extra_stems: StemMap = field(default_factory=dict)


@dataclass(frozen=True)
class StemChild:
    """The model and exact child output contributing to one parent stem."""

    model: object
    options: RunOptions = field(default_factory=lambda: RunOptions(kind=RunKind.SECONDARY))
    scale: float | None = None
    use_stem: str | None = None


@dataclass(frozen=True)
class SeparationBackend(Generic[Runtime]):
    """A family's state and typed functions; inference payloads stay private."""

    session: SeparationSession
    runtime: Runtime
    produce_stems: Callable[[SeparationSession, Runtime], ProducedStems]
    secondary_run_for: Callable[[SeparationSession, Runtime, str], StemChild | None] | None = None
