"""Secondary-model and vocal-chain workflow orchestration."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from gui_data.constants import DEMUCS_ARCH_TYPE, MDX_ARCH_TYPE, secondary_stem
from lib_v5 import spec_utils

if TYPE_CHECKING:
    from UVR import ModelData


def process_secondary_model(secondary_model: ModelData,
                            process_data,
                            main_model_primary_stem_4_stem=None,
                            is_source_load=False,
                            main_process_method=None,
                            is_pre_proc_model=False,
                            is_return_dual=True,
                            main_model_primary=None):
    """Run a configured secondary separator and normalize its return value."""
    if not is_pre_proc_model:
        process_data['process_iteration']()

    options = dict(
        main_model_primary_stem_4_stem=main_model_primary_stem_4_stem,
        main_process_method=main_process_method,
        main_model_primary=main_model_primary,
    )
    if secondary_model.process_method == DEMUCS_ARCH_TYPE or (
        secondary_model.process_method == MDX_ARCH_TYPE and secondary_model.is_mdx_c
    ):
        options['is_return_dual'] = is_return_dual

    # Lazy import avoids the factory/workflow/orchestration import cycle.
    from separation.factory import create_separator
    separator = create_separator(secondary_model, process_data, **options)
    secondary_sources = separator.seperate()

    if type(secondary_sources) is dict and not is_source_load and not is_pre_proc_model:
        return gather_sources(
            secondary_model.primary_model_primary_stem,
            secondary_stem(secondary_model.primary_model_primary_stem),
            secondary_sources,
        )
    return secondary_sources


def process_chain_model(secondary_model: ModelData,
                        process_data,
                        vocal_stem_path,
                        master_vocal_source,
                        master_inst_source=None):
    """Run the optional vocal-split or backing-vocal chain."""
    process_data['process_iteration']()

    if secondary_model.bv_model_rebalance:
        vocal_source = spec_utils.reduce_mix_bv(
            master_inst_source,
            master_vocal_source,
            reduction_rate=secondary_model.bv_model_rebalance,
        )
    else:
        vocal_source = master_vocal_source

    vocal_stem_path = [
        vocal_source,
        os.path.splitext(os.path.basename(vocal_stem_path))[0],
    ]

    from separation.factory import create_separator
    separator = create_separator(
        secondary_model,
        process_data,
        vocal_stem_path=vocal_stem_path,
        master_inst_source=master_inst_source,
        master_vocal_source=master_vocal_source,
    )
    secondary_sources = separator.seperate()
    return secondary_sources if type(secondary_sources) is dict else None


def gather_sources(primary_stem_name, secondary_stem_name, secondary_sources: dict):
    """Select the requested primary and secondary arrays from a stem mapping."""
    source_primary = False
    source_secondary = False

    for key, value in secondary_sources.items():
        if key in primary_stem_name:
            source_primary = value
        if key in secondary_stem_name:
            source_secondary = value

    return source_primary, source_secondary
