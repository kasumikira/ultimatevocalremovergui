"""Turn named model outputs into explicit audio files ready to persist."""

from __future__ import annotations

from gui_data.constants import (
    BV_VOCAL_STEM, BV_VOCAL_STEM_I, BV_VOCAL_STEM_LABEL,
    INFERENCE_STEP_DEVERBING, INST_STEM,
    LEAD_VOCAL_STEM, LEAD_VOCAL_STEM_I, LEAD_VOCAL_STEM_LABEL,
    VOCAL_STEM,
)
from lib_v5 import spec_utils

from .output import AudioArtifact, AudioOutput
from .state import RunKind


def prepare_output(session, stems: dict) -> list[AudioArtifact]:
    """Apply output policy without writing files or mutating the session."""
    artifacts: list[AudioArtifact] = []
    is_vocal_split = session.context.kind is RunKind.VOCAL_SPLIT
    standalone = not session.context.ensemble.shared_output or is_vocal_split

    def append(path, name, audio):
        should_deverb = session.output_policy.deverb_vocals and (
            session.deverb_vocal_opt == name
            or (session.deverb_vocal_opt == 'ALL' and name in (
                VOCAL_STEM, LEAD_VOCAL_STEM_LABEL, BV_VOCAL_STEM_LABEL)))
        if should_deverb and standalone:
            from .inference.vr import vr_denoiser
            session.services.write_to_console(
                INFERENCE_STEP_DEVERBING, base_text='')
            dry, reverb = vr_denoiser(
                audio, session.device_state.torch_device, is_deverber=True,
                model_path=session.DEVERBER_MODEL)
            artifacts.extend((
                AudioArtifact(path.replace('.wav', '_deverbed.wav'),
                              f'{name} (Deverbed)', dry),
                AudioArtifact(path.replace('.wav', '_reverb_only.wav'),
                              f'{name} (Reverb Only)', reverb),
            ))
        artifacts.append(AudioArtifact(path, name, audio))

    def append_split_vocal(name, audio):
        label = (LEAD_VOCAL_STEM_LABEL if name == LEAD_VOCAL_STEM
                 else BV_VOCAL_STEM_LABEL)
        append(session.audio_file_base_voc_split(VOCAL_STEM, name), label, audio)

    def append_split_instrumental(name, audio, invert=False):
        label = ('Instrumental (With Lead Vocals)'
                 if name == LEAD_VOCAL_STEM
                 else 'Instrumental (With Backing Vocals)')
        suffix = (LEAD_VOCAL_STEM_I if name == LEAD_VOCAL_STEM
                  else BV_VOCAL_STEM_I)
        contribution = -audio if invert else audio
        mixed = spec_utils.combine_arrarys(
            [session.vocal_chain.master_instrumental, contribution],
            is_swap=True)
        append(session.audio_file_base_voc_split(INST_STEM, suffix), label, mixed)

    for stem_name, audio in stems.items():
        model_lead = (session.output_policy.backing_vocal_rebalanced
                      and is_vocal_split and stem_name == LEAD_VOCAL_STEM)
        rebalance_backing = (
            session.output_policy.backing_vocal_rebalanced
            and is_vocal_split and stem_name == BV_VOCAL_STEM)
        skip_inst = (session.output_policy.save_vocal_only
                     and session.output_policy.secondary_bv_rebalance
                     and stem_name == INST_STEM)
        if rebalance_backing:
            master = spec_utils.match_array_shapes(
                session.vocal_chain.master_vocal, audio, is_swap=True)
            lead = audio - master
        if model_lead or skip_inst:
            continue
        if session.context.kind not in (RunKind.PRIMARY, RunKind.VOCAL_SPLIT):
            continue

        raw_path = AudioOutput().stem_path(session, stem_name)
        if is_vocal_split and not session.output_policy.split_instrumental_only:
            append_split_vocal(stem_name, audio)
            if rebalance_backing:
                append_split_vocal(LEAD_VOCAL_STEM, lead)
        elif not (session.output_policy.split_instrumental_only
                  and stem_name in (VOCAL_STEM, BV_VOCAL_STEM,
                                    LEAD_VOCAL_STEM)):
            append(raw_path, stem_name, audio)

        if (session.output_policy.save_split_instrumental
                and not session.output_policy.save_vocal_only):
            append_split_instrumental(stem_name, audio)
            if rebalance_backing:
                append_split_instrumental(LEAD_VOCAL_STEM, lead, invert=True)
    return artifacts
