# Separation responsibilities

The `separation` package is the public entry point for the `Seperate*` classes,
factory, audio-format helper and orchestration functions. Backend `seperate()`
workflows live directly under the package; model loading and numerical inference
live in `inference/`. Secondary-model orchestration and vocal chains live in
`orchestration.py`. `separate.py` remains an untouched reference and is never
imported by the replacement.

This package provides concrete operations used by those workflows:

| Module | Responsibility |
| --- | --- |
| `inference/mdxc.py` | MDXC architecture construction, strict checkpoint loading and overlap-add inference |
| `inference/mdx.py` | MDX settings, spectral inference and model execution |
| `inference/demucs.py` | Demucs inference |
| `inference/vr.py` | VR spectral preparation, inference, denoising and waveform reconstruction |
| `audio.py` | Audio loading, format conversion, pitch and denoising utilities |
| `output.py` | Stem writing, naming and output transformations |
| `state.py` | Initialization of the existing job/configuration state |
| `orchestration.py` | Secondary separators and vocal-split chain dispatch |
| `workflow.py` | Shared cache, progress, pitch and post-processing policy |
| `mdx.py` | MDX model loading and separation workflow |
| `mdxc.py` | MDXC stem selection and separation workflow |
| `demucs.py` | Demucs model loading and separation workflow |
| `vr.py` | VR model loading and separation workflow |

Implementation mixins preserve inherited method names and signatures without
duplicating forwarding methods. They operate on the existing job state and call
workflow hooks on `self`; they do not import the business module. This keeps the
dependency direction one-way without rewriting the GUI's `ModelData` contract.
They are not yet standalone inference services with independent request objects.

Model loading lifetime, numerical algorithms, output policy and cache behavior
are intentionally retained. In particular, overlap=1 retains the archived
zero-weight seam behavior, and recursive demudding still reloads MDXC models.

For an MDXC architecture with the existing constructor and forward contract,
import its constructor in `inference/mdxc.py` and add its type/section to `MDXC_MODELS`
there; GUI profiles are registered separately. Other inference contracts
need an explicit adapter. Do not replace existing architectures to add variants.

Each module explicitly imports its own dependencies. There is no shared import
aggregator or wildcard import; model constructors belong to `models.py`, while
the MDXC runner calls that module's factory without copying its namespace.

Tests: `uv run python -m unittest discover -s tests -p "test_*.py"`.
Coverage includes archive checksum, legacy signatures and method bodies, global
dependencies, model dispatch, strict loading, and numerical overlap-add comparison.
Real model weights, GPU backends and packaged application execution are not
covered by these CPU regression tests.
