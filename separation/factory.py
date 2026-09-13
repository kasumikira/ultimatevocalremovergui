"""Separator backend registration and construction."""

from separation.demucs import SeperateDemucs
from separation.mdx import SeperateMDX
from separation.mdxc import SeperateMDXC
from separation.vr import SeperateVR

BACKENDS = {
    'VR Arc': SeperateVR,
    'MDX-Net': SeperateMDX,
    'MDXC': SeperateMDXC,
    'Demucs': SeperateDemucs,
}


def create_separator(model_data, process_data, **options):
    """Construct the separator selected by a model configuration."""
    key = model_data.process_method
    if key == 'MDX-Net' and model_data.is_mdx_c:
        key = 'MDXC'
    try:
        constructor = BACKENDS[key]
    except KeyError:
        raise ValueError(f'Unknown separation backend: {key}') from None
    return constructor(model_data, process_data, **options)
