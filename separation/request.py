"""What one model run is given.

``RunRequest`` describes *what to run*: the audio, the model configuration and
where the run writes. ``RunServices`` carries the application side of a run: the
UI reporting callbacks and the job-level source cache. Keeping the two apart is
what lets a run be described, built and tested without dragging the GUI along.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from UVR import ModelData


class RunKind(Enum):
    PRIMARY = auto()
    SECONDARY = auto()
    PREPROCESS = auto()
    VOCAL_SPLIT = auto()


@dataclass(frozen=True)
class RunOptions:
    """The role of this invocation, independent of the model it uses."""

    kind: RunKind | None = None
    parent_stem: str | None = None
    target_stem: str | None = None
    parent_method: str | None = None
    wants_stem_list: bool = False
    master_instrumental: np.ndarray | None = field(default=None, compare=False)
    master_vocal: np.ndarray | None = field(default=None, compare=False)


class StemRole(Enum):
    PRIMARY = 'primary'
    SECONDARY = 'secondary'


ALL_STEM_ROLES = frozenset(StemRole)


@dataclass(frozen=True)
class StemSelection:
    """The concrete named stems requested from a model, or all of them."""

    names: frozenset[str] | None = None

    @classmethod
    def all(cls) -> StemSelection:
        return cls()

    @classmethod
    def only(cls, *names: str) -> StemSelection:
        if not names:
            raise ValueError('A stem selection must contain at least one name')
        return cls(frozenset(names))

    def apply(self, available: tuple[str, ...]) -> tuple[str, ...]:
        if self.names is None:
            return available
        missing = self.names.difference(available)
        if missing:
            raise ValueError(f'Requested stems are unavailable: {sorted(missing)}')
        return tuple(name for name in available if name in self.names)


def _stem_roles_from_legacy_flags(primary_only: bool,
                                  secondary_only: bool) -> frozenset[StemRole]:
    """Translate the old GUI switches once, as a request enters the core."""
    if primary_only and secondary_only:
        return frozenset()
    if primary_only:
        return frozenset({StemRole.PRIMARY})
    if secondary_only:
        return frozenset({StemRole.SECONDARY})
    return ALL_STEM_ROLES


def _report_nothing(*args, **kwargs):
    """Default service: report nothing, count nothing."""


def _no_cached_run(process_method, model_name=None):
    """Default service: this file has no earlier run of this model."""
    return None


def _runs_once(model_name) -> int:
    """Default service: every model is invoked once per audio file."""
    return 1


@dataclass(frozen=True)
class RunRequest:
    """One model run: what to run, on what, and where it writes.

    Deliberately small. Only values that change what the model computes live
    here; anything that only affects reporting or job bookkeeping belongs to
    :class:`RunServices`.
    """

    audio_file: str | np.ndarray
    export_path: str
    audio_file_base: str
    model: ModelData
    is_ensemble_master: bool = False
    is_4_stem_ensemble: bool = False
    #: Explicit public output selection. ``None`` reads the legacy GUI flags.
    stem_selection: StemSelection | None = None

    def requested_stem_roles(self, *, child_run: bool) -> frozenset[StemRole]:
        """Return the output roles requested by legacy model configuration."""
        if child_run:
            return ALL_STEM_ROLES
        return _stem_roles_from_legacy_flags(
            self.model.is_primary_stem_only,
            self.model.is_secondary_stem_only,
        )


@dataclass(frozen=True)
class RunServices:
    """The application side of a run: UI reporting and job bookkeeping.

    The defaults are inert, so a run can be constructed and executed without a
    GUI attached; a real run injects the GUI callbacks and the job cache.
    """

    set_progress_bar: Callable = _report_nothing
    write_to_console: Callable = _report_nothing
    process_iteration: Callable = _report_nothing
    load_cached_run: Callable = _no_cached_run
    store_cached_run: Callable = _report_nothing
    model_occurrences: Callable = _runs_once
