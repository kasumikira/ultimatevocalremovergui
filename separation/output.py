from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from gui_data.constants import DONE, SAVING_STEM
from lib_v5 import spec_utils

from .audio import save_format
from .state import RunKind


@dataclass(frozen=True)
class AudioArtifact:
    """One fully prepared audio file for the output sink."""

    path: str
    stem_name: str
    audio: np.ndarray
    samplerate: int = 44100


class AudioOutput:
    """Persist prepared audio artifacts without running separation models."""

    @staticmethod
    def stem_path(session, stem_name: str) -> str:
        return os.path.join(
            session.export_path, f'{session.audio_file_base}_({stem_name}).wav')

    def write_artifacts(self, session, artifacts: list[AudioArtifact]) -> None:
        standalone_output = (
            not session.context.ensemble.shared_output
            or session.context.kind is RunKind.VOCAL_SPLIT)
        for artifact in artifacts:
            session.services.write_to_console(
                f'{SAVING_STEM[0]}{artifact.stem_name}{SAVING_STEM[1]}')
            source = spec_utils.normalize(
                artifact.audio, session.output_policy.normalize)
            path = artifact.path
            if os.path.isfile(path) and standalone_output:
                path = path.replace(
                    '.wav', f'_{session.overwrite_protect_stamp}.wav')
            sf.write(path, source, artifact.samplerate,
                     subtype=session.wav_type_set)
            if standalone_output:
                save_format(path, session.save_format, session.mp3_bit_set)
            session.services.write_to_console(DONE, base_text='')
        if artifacts:
            session.services.set_progress_bar(0.95)
