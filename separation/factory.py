"""Select a backend lazily and execute it through the shared runner."""

from importlib import import_module

from separation.pipeline import SeparationRunner
from separation.request import RunOptions, RunRequest


BACKENDS = {
    'VR Arc': 'vr',
    'MDX-Net': 'mdx',
    'MDXC': 'mdxc',
    'Demucs': 'demucs',
}


def backend_constructor(model_data):
    """Resolve the registered constructor for a model configuration."""
    key = model_data.process_method
    if key == 'MDX-Net' and model_data.is_mdx_c:
        key = 'MDXC'
    try:
        module_name = BACKENDS[key]
    except KeyError:
        raise ValueError(f'Unknown separation backend: {key}') from None
    return import_module(f'separation.inference.{module_name}').create_backend


def create_separator(request: RunRequest, services, options: RunOptions | None = None):
    """Compose the selected backend with the shared runner."""
    backend = backend_constructor(request.model)(request, services, options)
    return SeparationRunner(backend)


def run_separator(request: RunRequest, services, options: RunOptions | None = None):
    """Run one backend and return its normalized result.

    This is the only entry point the caller needs: it never imports a concrete
    separator class and never touches its internal state.
    """
    return create_separator(request, services, options).run()
