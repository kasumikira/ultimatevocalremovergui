# Separation

`UVR.py` submits a `RunRequest` and `RunServices` to `run_separator`. The result
always contains named audio, including for primary runs. Saving is a separate
policy: secondary and preprocessing runs return audio without writing files.
`separate.py` is an unchanged reference for numerical regression tests.

## Execution

The shared runner executes these steps in order:

1. Run the backend, resolve selection against its actual stems, and validate the result.
2. Run the configured secondary models and blend their exact named outputs.
3. Prepare export artifacts, including optional deverb and vocal-split mixes.
4. Save the prepared artifacts.
5. Run the optional vocal chain once, on the completed primary result.

Unused accelerator cache is cleared in `finally`. Loaded models belong to the
backend and are released there, before secondary models and output processing.
Demucs preprocessing is an explicit backend dependency: it runs a child with
`RunKind.PREPROCESS` to obtain the mixture required by Demucs inference. It is
not launched by file writing or by an output callback.

## Configuration

`RunOptions` describes an invocation's role, parent/target stem and optional
vocal-chain source audio. Child dispatch always supplies its role explicitly;
it does not mutate the model's GUI flags. A top-level call may omit options,
in which case legacy model flags are interpreted at construction.

`workflow.create_run` translates shared GUI settings, passes complete typed
`BackendSettings` to the selected family's setup function, aligns legacy
ensemble selection and constructs a fully initialized `SeparationSession`.
Family setup returns updated settings using `dataclasses.replace`, together
with its inference runtime. There are no string attribute paths, generic
state patches, or partially initialized session objects.

Stem, device, ensemble and output settings are immutable. Backends must choose
stem names during setup; deriving a complement must not rename shared stems
while another stage is using them. Mutable inference data stays in the
backend's runtime; only progress and vocal-chain bookkeeping change in the
session.

MDX-Net and MDXC have independent runtimes and configuration adapters.
`MDXRuntime` contains ONNX/checkpoint and STFT state; `MDXCRuntime` contains
config-driven network, overlap-add and multi-stem state. MDXC keeps one
`config` reference and does not import the MDX-Net backend. Shared services,
audio utilities and denoising remain shared without a common runtime base.

The existing GUI `ModelData` is still the configuration source at this boundary.
Family setup translates its family-specific fields; this change does not
replace the GUI model catalog or model download/configuration handling.

## Audio contract

Every value in `ProducedStems.stems` and `SeparationResult.stems` is a NumPy
array shaped `(samples, 2)` at **44100 Hz**. Inference may use different shapes
internally, but converts before returning. The backend resolves selection after inference, so Demucs six-source output
is recognized even when initial metadata lists only four sources. The runner
checks explicit named selections and array shapes before saving.

`StemSelection.only('Vocals', 'Drums')` selects named outputs; `None` on the
request translates the existing GUI stem-only settings. Child runs request
the audio their caller needs independently of the GUI save selection.

A `StemChild` specifies a model, typed invocation options, blend scale and exact
child stem name. No substring matching or backend-specific array indices cross
this boundary. A per-stem secondary policy owns its ordinary pair case too;
returning `None` unambiguously means no child for that stem. Identical child
invocations are reused within one blend step.

`ProducedStems.extra_stems` is reserved for separately named export derivatives
such as the Demucs preprocessed instrumental. These do not launch secondary
models or vocal chains and do not appear in the returned stem map. Regular
multi-stem output belongs in `stems` for every backend.

## Output and caching

`postprocess.prepare_output` applies saving policy and prepares `AudioArtifact`
objects. It performs optional deverb inference and vocal-split mixing but does
not write files or mutate the session. `AudioOutput.write_artifacts` performs
normalization, overwrite protection, SoundFile writing and format conversion;
it never starts a model or a vocal chain.

The per-file cache stores opaque inference payloads by architecture and model
basename through `RunServices`. Only the family reads its payload. Cache entries
must not be mutated during stem derivation. Vocal-chain runs use different
input audio, so they neither read nor populate the mixture cache. Services are
provided by the GUI; the separation core does not depend on a GUI window.

## Files and tests

| File | Responsibility |
| --- | --- |
| `request.py` | Request, invocation options, named selection, application callbacks |
| `state.py` | Typed settings and legacy selection alignment |
| `workflow.py` | Session construction, inference reporting, per-file cache and pitch helpers |
| `contract.py` | Typed backend functions, named audio and secondary contribution |
| `factory.py` | Backend selection and run entry point |
| `pipeline.py` | Ordered execution and child dispatch |
| `postprocess.py` | Prepare audio artifacts according to output policy |
| `output.py` | Save prepared files |
| `inference/` | Family setup, model loading, inference and stem derivation |

Run `uv run python -m unittest discover -s tests -p "test_*.py"`.
Tests mock models and file writes, verify orchestration behavior, and compare
selected numerical paths against the archived implementation. They do not
launch Tkinter, download models or prove GPU performance/real-model quality.
