# Official UVR pyc comparison

Audit date: 2026-09-05 (Asia/Shanghai)

This document records the final comparison between the current repository and
the Python 3.9 bytecode extracted from the official UVR executable
`UVR_Patch_4_24_25_20_11_BETA` (UVR 5.6.1).

This is a frozen record of restoration commit
`b83971f0d685032225204d33bc66e598056b6c83` (`restore from official pyc code`).
Later commits intentionally diverge from the official executable and are not
reflected here.

## Baseline and method

Official bytecode root:

```text
D:\Download\UVR\UVR.exe_extracted\PYZ-00.pyz_extracted
```

Repository commit at audit time:

```text
b83971f0d685032225204d33bc66e598056b6c83
```

The aggregate SHA-256 below covers the 44 comparable `lib_v5` Python sources
plus `separate.py` in that commit, sorted by repository path and framed with
each path:

```text
source files: 45
source aggregate SHA-256: 02948f867bae3198c7a17914a86eab2562d30aad9144487c5e6789ee236da6fa
```

The official comparison set contains `lib_v5.pyc`, all 44 pycs below its
`lib_v5` directory, and `separate.pyc`:

```text
official pyc files: 46
official aggregate SHA-256: dc8c053e5022a15d0abe2bed1d144599adab4375bdca93dfc3bd9676ee647a61
```

Each source file was compiled with CPython 3.9 at optimization level 0. Every
code object was then matched recursively by its module/class/function nesting
and compared on:

- opcode and resolved instruction arguments;
- positional, keyword-only, and local-variable layouts;
- referenced global/attribute names;
- non-code constants;
- nested code-object inventory.

Filename, first-line number, line table, pyc header, and source formatting were
not treated as semantic differences. Consequently, “semantic match” below does
not mean that recompiling the source produces a byte-for-byte identical pyc.

## Summary

| Result | Count | Meaning |
| --- | ---: | --- |
| Semantic match | 37 | No detected executable code-object difference |
| Difference | 8 | Listed individually below |
| Packaging-only items | 2 | No one-to-one `lib_v5` source/pyc path |

`separate.py` is included in the 37 semantic matches. No missing or extra
functions, methods, lambdas, comprehensions, or nested code objects were found
in any of the 45 comparable modules.

## Files with differences

### 1. `lib_v5/spec_utils.py`

Affected functions:

- `apply_phase_from_audio`
- `reshape_if_needed`
- `transfer_phase_only`

Difference type: docstring whitespace only.

The executable instructions, parameters, local variables, referenced names,
and all non-docstring constants match. Blank lines inside these three
docstrings contain different spaces. This changes `function.__doc__` but not
audio processing.

Impact: none on model loading, inference, or generated audio.

### 2. `lib_v5/bandit/core/model/bsrnn/maskestim.py`

Affected constructors:

- `BaseNormMLP.__init__`
- `NormMLP.__init__`
- `MultAddNormMLP.__init__`

Current source retains the upstream wrappers:

```python
torch.jit.script(nn.Sequential(...))
```

The official pyc constructs the same `nn.Sequential(...)` objects without
calling `torch.jit.script`.

Impact:

- layer dimensions and state-dict key layout are unchanged;
- the current modules are scripted during construction, whereas official UVR
  leaves them as ordinary eager `nn.Sequential` modules;
- construction can therefore differ in JIT compatibility, startup cost, and
  module type, even though the contained layers and forward computation are the
  same.

This is an intentional upstream difference, not a decompilation uncertainty.

### 3. `lib_v5/bandit/core/model/bsrnn/tfmodel.py`

Affected constructor:

- `ConvolutionalTimeFreqModule.__init__`

Current source applies `torch.jit.script` to `self.seqband`; official UVR stores
the same `nn.Sequential` directly.

Impact: the same JIT/eager distinction described for `maskestim.py`. The
sequence contents and checkpoint keys are unchanged.

### 4. `lib_v5/bandit_v2/bandit.py`

Affected code object:

- class body `Bandit`

The executable methods, including `Bandit.__init__` and `Bandit.forward`, match.
Only four constructor annotations were restored to the upstream Python 3.10
form:

```diff
-hidden_activation_kwargs: Optional[Dict] = None
-win_length: Optional[int] = 2048
-wkwargs: Optional[Dict] = None
-power: Optional[int] = None
+hidden_activation_kwargs: Dict | None = None
+win_length: int | None = 2048
+wkwargs: Dict | None = None
+power: int | None = None
```

Impact: `Bandit.__annotations__`/signature metadata differs. Model construction,
state dicts, and forward calculation do not. The current form intentionally
requires Python 3.10 or newer when the class body is executed.

### 5. `lib_v5/roformer/attend_new.py`

Affected constructor:

- `Attend.__init__`

Current source follows lucidrains upstream and prints one additional diagnostic
on non-A100 CUDA GPUs:

```python
print_once(
    'Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda'
)
```

The official pyc sets the same `FlashAttentionConfig(False, True, True)` but
does not print this message.

Impact: console output only. Attention backend selection and numerical behavior
are unchanged.

Official pyc SHA-256:

```text
f1d9e0fab8f4d4954dbdaed7c90d82ec9215e4a641bb9ad874c2261a730f5c49
```

### 6. `lib_v5/roformer/attend.py`

Affected code objects:

- module body
- `Attend.__init__`

Current source imports `os` and generalizes CUDA backend selection. The official
pyc uses Flash Attention only when the CUDA compute capability is exactly 8.0
(A100); every other CUDA capability receives math/memory-efficient attention.

Official behavior, simplified:

```python
if device_properties.major == 8 and device_properties.minor == 0:
    cuda_config = FlashAttentionConfig(True, False, False)
else:
    cuda_config = FlashAttentionConfig(False, True, True)
```

Current behavior, simplified:

```python
if compute_capability >= 8.0:
    if os.name == 'nt':
        cuda_config = FlashAttentionConfig(False, True, True)
    else:
        cuda_config = FlashAttentionConfig(True, False, False)
else:
    cuda_config = FlashAttentionConfig(False, True, True)
```

Practical differences:

- Windows A100: official enables Flash Attention; current source disables it.
- Non-Windows Ampere/Ada GPUs with capability above 8.0: official disables
  Flash Attention; current source enables it.
- CPUs, unavailable CUDA, `flash=False`, and CUDA capability below 8.0 follow
  the same backend class.

The attention equations, scaling, dropout, and non-flash path match. This can
change performance, memory usage, and small floating-point details, but does
not change model parameters or checkpoint compatibility.

Official pyc SHA-256:

```text
8c4932bb030aac07ac721766a8a6404dec8ef33b55df0593472bbf530e0e0fc5
```

### 7. `lib_v5/roformer/bs_roformer_new.py`

Affected code objects:

- module body
- class bodies `BandSplit`, `MaskEstimator`, and `BSRoformer`

No constructor or forward method has an instruction difference. The class-body
differences come from restoring the upstream author's Python 3.10 annotations:

- `from __future__ import annotations` is present;
- `tuple[int, ...]` replaces `Tuple[int, ...]`;
- `Callable | None` replaces `Optional[Callable]`;
- no otherwise-unused `Tuple`, `Optional`, or `List` aliases are imported from
  `beartype.typing`.

Impact: annotation storage and resolution differ, including what the
`@beartype` decorator initially receives. With the supported Python 3.10 and
current beartype dependency these forms describe the same types. Model graph,
parameter names, defaults, and forward logic match the official pyc.

### 8. `lib_v5/roformer/mel_band_roformer_new.py`

Affected code objects:

- module body
- class bodies `BandSplit`, `MaskEstimator`, and `MelBandRoformer`

The difference is the same Python 3.10 annotation restoration described for
`bs_roformer_new.py`. Every constructor and forward method matches at the
instruction level.

Impact: annotation/decorator metadata only; no model graph, checkpoint, or
audio-path difference was detected.

## Semantically matching files

The following files have matching recursive code-object inventories and no
detected semantic differences:

```text
separate.py

lib_v5/apollo_inference.py
lib_v5/bypass_check_patch.py
lib_v5/mdxnet.py
lib_v5/modules.py
lib_v5/pyrb.py
lib_v5/tfc_tdf_v3.py
lib_v5/verify_gpu_availability.py

lib_v5/apollo_model_data/__init__.py
lib_v5/apollo_model_data/apollo.py
lib_v5/apollo_model_data/base_model.py

lib_v5/bandit/__init__.py
lib_v5/bandit/core/__init__.py
lib_v5/bandit/core/model/__init__.py
lib_v5/bandit/core/model/_spectral.py
lib_v5/bandit/core/model/bsrnn/__init__.py
lib_v5/bandit/core/model/bsrnn/bandsplit.py
lib_v5/bandit/core/model/bsrnn/core.py
lib_v5/bandit/core/model/bsrnn/utils.py
lib_v5/bandit/core/model/bsrnn/wrapper.py

lib_v5/bandit_v2/__init__.py
lib_v5/bandit_v2/bandsplit.py
lib_v5/bandit_v2/maskestim.py
lib_v5/bandit_v2/tfmodel.py
lib_v5/bandit_v2/utils.py

lib_v5/roformer/__init__.py
lib_v5/roformer/bs_roformer.py
lib_v5/roformer/mel_band_roformer.py

lib_v5/scnet/__init__.py
lib_v5/scnet/scnet.py
lib_v5/scnet/separation.py

lib_v5/vr_network/__init__.py
lib_v5/vr_network/layers.py
lib_v5/vr_network/layers_new.py
lib_v5/vr_network/model_param_init.py
lib_v5/vr_network/nets.py
lib_v5/vr_network/nets_new.py
```

The exact `separate.pyc` used in this comparison has SHA-256:

```text
6f868b3327d9e9c70e348cfb2405e0624d47c174438853879d3419b94deaf3b1
```

## Packaging-only differences

### `lib_v5.pyc`

The executable contains an empty `lib_v5.pyc` package initializer. The current
tree uses the `lib_v5/` directory as a namespace package and has no
`lib_v5/__init__.py`. The official initializer contains only `return None`, so
this layout difference adds no initialization behavior.

### `lib_v5/results.py`

There is no official `lib_v5/results.pyc`. The current file is an extra,
currently unreferenced copy of Matchering's results helpers. Its code objects
match the official `matchering/results.pyc`, but its package path is different.

Impact: none unless another module explicitly imports `lib_v5.results`.

## Overall conclusion

The restored separation and model code is very close to the official
executable:

- `separate.py` and 36 `lib_v5` modules are semantic matches;
- no function or method is missing or extra in the comparable modules;
- the only detected audio-runtime difference is the old RoFormer CUDA attention
  backend policy;
- Bandit Plus retains four upstream JIT wrappers that alter eager-versus-script
  construction but not layer/checkpoint layout;
- New RoFormer and Bandit v2 differences are intentional Python 3.10 annotation
  restorations;
- `attend_new.py` adds only a diagnostic print;
- `spec_utils.py` differs only in docstring whitespace.

This report establishes code-object equivalence, not numerical identity across
PyTorch versions, GPU backends, FFT implementations, or different dependency
sets. No model inference or audio golden-file comparison was performed as part
of this final audit.
