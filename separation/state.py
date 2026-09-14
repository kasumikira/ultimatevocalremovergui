from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, auto

import torch

from lib_v5.verify_gpu_availability import (
    GPU_TYPE_CPU,
)
from separation.request import (
    ALL_STEM_ROLES,
    RunKind,
    RunRequest,
    StemRole,
    StemSelection,
)

CPU_DEVICE = torch.device('cpu')


@dataclass(frozen=True)
class EnsembleContext:
    shared_output: bool
    multi_stem_output: bool
    per_stem_routing: bool


@dataclass(frozen=True)
class RunContext:
    kind: RunKind
    ensemble: EnsembleContext


def classify_run(model_data) -> RunKind:
    """Classify one model invocation into exactly one orchestration role."""
    if model_data.is_pre_proc_model:
        return RunKind.PREPROCESS
    if model_data.is_vocal_split_model:
        return RunKind.VOCAL_SPLIT
    if model_data.is_secondary_model:
        return RunKind.SECONDARY
    return RunKind.PRIMARY


class StemAlignment(Enum):
    NONE = auto()
    ENSEMBLE_PAIR = auto()
    TARGET_ONLY = auto()


@dataclass(frozen=True)
class StemState:
    primary: str
    secondary: str
    ensemble_primary: str | None
    legacy_requested_roles: frozenset[StemRole]
    selection: StemSelection | None
    alignment: StemAlignment = StemAlignment.NONE
    target_instrument: bool = False

    def select(self, available: tuple[str, ...]) -> tuple[str, ...]:
        """Resolve names against the model's actual output, including dynamic stems."""
        if self.selection is not None:
            return self.selection.apply(available)
        if len(available) > 2:
            return available
        by_role = {StemRole.PRIMARY: self.primary, StemRole.SECONDARY: self.secondary}
        return tuple(by_role[role] for role in StemRole
                     if role in self.legacy_requested_roles and by_role[role] in available)


@dataclass(frozen=True)
class PitchPolicy:
    enabled: bool
    semitones: float
    match_frequency: bool


@dataclass(frozen=True)
class SecondaryPlan:
    model: object
    scale: float | None


@dataclass(frozen=True)
class DeviceState:
    requested_device: object
    torch_device: object = CPU_DEVICE
    gpu_type: str = GPU_TYPE_CPU
    onnx_providers: tuple[str, ...] = ('CPUExecutionProvider',)


@dataclass
class ProgressState:
    completed_steps: int = 0


@dataclass
class VocalChain:
    model: object
    master_instrumental: object = None
    master_vocal: object = None
    master_vocal_path: str | None = None


@dataclass(frozen=True)
class OutputPolicy:
    normalize: bool
    deverb_vocals: bool
    save_split_instrumental: bool
    split_instrumental_only: bool
    save_vocal_only: bool
    secondary_bv_rebalance: bool
    karaoke: bool
    backing_vocal_model: bool
    backing_vocal_rebalanced: bool
    save_preprocessed_instrumental: bool


@dataclass(frozen=True)
class BackendSettings:
    """Complete settings chosen before constructing a session."""

    stems: StemState
    device: DeviceState
    ensemble: EnsembleContext


def align_stem_selection(settings: BackendSettings, request: RunRequest,
                         kind: RunKind) -> BackendSettings:
    stems = settings.stems
    roles = stems.legacy_requested_roles
    policy_applies = (
        stems.alignment is StemAlignment.ENSEMBLE_PAIR
        and (stems.target_instrument or not settings.ensemble.multi_stem_output)
    ) or (stems.alignment is StemAlignment.TARGET_ONLY and stems.target_instrument)
    if (request.is_ensemble_master and policy_applies
            and stems.ensemble_primary != stems.primary):
        roles = frozenset(StemRole.SECONDARY if role is StemRole.PRIMARY
                          else StemRole.PRIMARY for role in roles)
    # Child invocations supply audio to their caller, not the GUI's save selection.
    selection = stems.selection
    if kind is not RunKind.PRIMARY:
        roles, selection = ALL_STEM_ROLES, StemSelection.all()
    return replace(settings, stems=replace(
        stems, legacy_requested_roles=roles, selection=selection))
