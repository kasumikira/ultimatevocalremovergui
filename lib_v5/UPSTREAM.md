# `lib_v5` upstream diff

This document compares every restored architecture file with the exact upstream
revision used as its source anchor. The restored files have been reformatted to
remove known blank-line and final-newline noise; the differences below are the
remaining source differences, not a normalized view that hides formatting.

The comparison was made directly against Git objects, not against the current
upstream working trees.

| UVR architecture | Exact upstream anchor |
| --- | --- |
| Bandit Plus | ZFTurbo/Music-Source-Separation-Training `9fbe4b6b6908ee521df9bd9a3632bc562093a4f1` (2023-12-19) |
| Bandit v2 | ZFTurbo/Music-Source-Separation-Training `4918d9810e4b4238b98c00046c3f3f9169e0d246` (2024-07-20) |
| SCNet | ZFTurbo/Music-Source-Separation-Training `7c4d8f908b5979645ab04b2a2cbc974318fa9b3c` (2024-07-10) |
| BS/Mel RoFormer (old) | ZFTurbo/Music-Source-Separation-Training `2b6f12dfdb0362a7b510c3786891cc65bcc91070` (2025-03-27) |
| BS/Mel RoFormer (new) | lucidrains/BS-RoFormer `87576b9096ea6dcef9e340745d87d72f30bdd985` (2024-12-22) |

`-` means the anchored upstream source and `+` means the restored UVR source.

## Bandit Plus

Anchor: MSST `9fbe4b6`.

### `bandit/core/model/__init__.py`

No source diff.

### `bandit/core/model/_spectral.py`

No source diff.

### `bandit/core/model/bsrnn/__init__.py`

Only the package prefix changed:

```diff
-from models.bandit.core.model.bsrnn.bandsplit import BandSplitModule
-from models.bandit.core.model.bsrnn.tfmodel import (...)
+from lib_v5.bandit.core.model.bsrnn.bandsplit import BandSplitModule
+from lib_v5.bandit.core.model.bsrnn.tfmodel import (...)
```

### `bandit/core/model/bsrnn/bandsplit.py`

Only the package prefix changed:

```diff
-from models.bandit.core.model.bsrnn.utils import (...)
+from lib_v5.bandit.core.model.bsrnn.utils import (...)
```

### `bandit/core/model/bsrnn/core.py`

Only the four internal imports changed from `models.bandit...` to
`lib_v5.bandit...`. All classes and function bodies are unchanged.

### `bandit/core/model/bsrnn/maskestim.py`

Only the `utils` import changed from `models.bandit...` to `lib_v5.bandit...`.
The upstream `torch.jit.script(nn.Sequential(...))` wrappers in
`BaseNormMLP.__init__`, `NormMLP.__init__`, and `MultAddNormMLP.__init__` are
retained. This is intentionally different from the official pyc, which does
not contain those wrappers, but it does not alter checkpoint keys.

### `bandit/core/model/bsrnn/tfmodel.py`

No source diff. This includes the upstream `torch.jit.script` wrapper in
`ConvolutionalTimeFreqModule.__init__`; the official pyc omits that wrapper.

### `bandit/core/model/bsrnn/utils.py`

No source diff.

### `bandit/core/model/bsrnn/wrapper.py`

Only two internal imports changed from `models.bandit...` to
`lib_v5.bandit...`. `BaseBandit`, `MultiMaskMultiSourceBandSplitRNNSimple`, and
their methods are otherwise identical to the anchor.

### Package placeholders

`bandit/__init__.py` and `bandit/core/__init__.py` are empty UVR package
placeholders. They contain no model logic.

## Bandit v2

Anchor: MSST `4918d98`.

### `bandit_v2/bandit.py`

No source diff. The upstream author's PEP 604 union annotations are retained;
this restored tree targets Python 3.10 or newer rather than Python 3.9.

### `bandit_v2/bandsplit.py`

No source diff.

### `bandit_v2/maskestim.py`

No source diff.

### `bandit_v2/tfmodel.py`

No source diff.

### `bandit_v2/utils.py`

No source diff.

### `bandit_v2/__init__.py`

Empty UVR package placeholder; no model logic.

## SCNet

Anchor: MSST `7c4d8f9`.

### `scnet/scnet.py`

Changed symbol: `SCNet.forward`.

```diff
+is_using_other_gpu = lambda device: not (
+    device == 'cpu' or device.startswith('cuda')
+)

 def SCNet.forward(self, x):
+    original_device = x.device
+    is_open_cl = 'privateuseone' in original_device.type
+    is_other_gpu = is_using_other_gpu(x.device.type)
+    if is_other_gpu:
+        x = x.cpu()
     # torch.stft / complex conversion
+    if is_other_gpu:
+        x = x.to(original_device)
     # network body
+    if is_other_gpu:
+        x = x.cpu()
     # complex conversion / torch.istft
+    if is_other_gpu:
+        x = x.to(original_device)
```

Purpose: FFT and complex-tensor conversion run on CPU for devices other than
CPU and CUDA (notably MPS and DirectML), while the neural network body remains
on the original device. `is_open_cl` is assigned because it exists in the
official bytecode, although this function does not subsequently read it.

### `scnet/separation.py`

Changed symbol: `FeatureConversion.forward`.

```diff
+is_using_other_gpu = lambda device: not (
+    device == 'cpu' or device.startswith('cuda')
+)

 def FeatureConversion.forward(self, x):
+    original_device = x.device
+    is_open_cl = 'privateuseone' in original_device.type
+    is_other_gpu = is_using_other_gpu(x.device.type)
+    if is_other_gpu:
+        x = x.cpu()
     # view_as_complex / view_as_real conversion
+    if is_other_gpu:
+        x = x.to(original_device)
```

Again, `is_open_cl` is an official-bytecode local that is assigned but unused.

### `scnet/__init__.py`

Empty UVR package placeholder; no model logic.

## BS-RoFormer and Mel-Band RoFormer (old)

Anchor: MSST `2b6f12d`. These are the non-value-residual architectures used by
the corresponding official UVR models.

### `roformer/attend.py`

Changed symbols: `Attend.__init__`, `Attend.forward`, and `Attend.flash_attn`.
UVR removed configurable attention scaling from this old architecture:

```diff
-def default(v, d):
-    return v if exists(v) else d
-
-def __init__(self, dropout=0., flash=False, scale=None):
+def __init__(self, dropout=0., flash=False):
     ...
-    self.scale = scale

-if exists(self.scale):
-    default_scale = q.shape[-1] ** -0.5
-    q = q * (self.scale / default_scale)

-scale = default(self.scale, q.shape[-1] ** -0.5)
+scale = q.shape[-1] ** -0.5
```

The anchor's CUDA capability/Flash-Attention selection code is otherwise
retained. The current generalized A100 handling is therefore inherited from
this MSST anchor, not an additional UVR change.

### `roformer/bs_roformer.py`

Changed symbols: `RMSNorm.forward`, `Attention.forward`,
`BSRoformer.__init__`, and `BSRoformer.forward`.

```diff
-from models.bs_roformer.attend import Attend
+from .attend import Attend

 def RMSNorm.forward(self, x):
+    x = x.to(self.gamma.device)
     return F.normalize(x, dim=-1) * self.scale * self.gamma

 def Attention.forward(...):
+    original_device = x.device
+    is_open_cl = 'privateuseone' in original_device.type
+    if is_open_cl:
+        q, k = q.cpu(), k.cpu()
     # apply rotary embedding
+    if is_open_cl:
+        q, k = q.to(original_device), k.to(original_device)
```

Constructor probe:

```diff
-torch.stft(torch.randn(1, 4096), ..., window=torch.ones(stft_win_length))
+torch.stft(torch.randn(1, 4096), ...)
```

Forward/device path:

```diff
+original_device = raw_audio.device
+x_is_mps = original_device.type == 'mps' or 'privateuseone' in original_device.type
+if x_is_mps:
+    raw_audio = raw_audio.cpu()

-try:
-    stft_repr = torch.stft(...)
-except:
-    stft_repr = torch.stft(cpu_inputs...).to(device)
+stft_repr = torch.stft(cpu_or_original_inputs...)

+if x_is_mps:
+    mask = mask.cpu()

-recon_audio = torch.istft(..., length=raw_audio.shape[-1])
+recon_audio = torch.istft(...)

+# Move training losses back to original_device when the CPU FFT path was used.
```

The important behavioral changes are deterministic CPU FFT handling for MPS
and DirectML, removal of the explicit ISTFT `length`, and loss-device repair.

### `roformer/mel_band_roformer.py`

Changed symbols: `RMSNorm.forward`, `Attention.forward`,
`MaskEstimator.__init__`, and `MelBandRoformer.forward`; added top-level
functions: `log_weighted_mae` and `l2_freq_loss`.

The RMSNorm and DirectML rotary changes are the same as in
`bs_roformer.py`. The internal import is also rewritten to `.attend`.

```diff
 def MaskEstimator.__init__(...):
-    net = []  # redundant assignment before the loop
     for dim_in in dim_inputs:
         net = []

+def log_weighted_mae(recon_mag, target_mag, weight=None, eps=1e-8): ...
+def l2_freq_loss(recon_mag, target_mag, eps=1e-8): ...
```

The forward path adds the same MPS/DirectML CPU-FFT dispatch as the BS model,
plus CPU placement for every tensor participating in frequency indexing and
mask reconstruction:

```diff
+freq_indices = self.freq_indices.cpu() if x_is_mps else self.freq_indices
+if x_is_mps:
+    masks = masks.cpu()
+    scatter_indices = repeat(self.freq_indices.cpu(), ...)
+    denom = denom.cpu()
+# Move total and component losses back to original_device before returning.
```

## BS-RoFormer New and Mel-Band RoFormer New

Anchor: lucidrains/BS-RoFormer `87576b9`. These files intentionally preserve
the anchor's value-residual implementation. They do **not** use MSST's later
Hyper Connections experimental architecture, whose graph and checkpoint keys
are incompatible with the official executable.

### `roformer/attend_new.py`

No source diff.

### `roformer/bs_roformer_new.py`

Changed symbols: `RMSNorm.forward`, `Attention.forward`,
`BSRoformer.__init__`, and `BSRoformer.forward`.
Added symbols: `l2norm` and `LinearAttention`.

Compatibility/import changes:

```diff
-from bs_roformer.attend import Attend
+from .attend_new import Attend
+from einops.layers.torch import Rearrange
```

The upstream author's `from __future__ import annotations`, built-in generic
annotations, and PEP 604 unions are retained for the Python 3.10 target.

UVR adds the `l2norm` helper and a complete `LinearAttention` class. In this BS
variant the class is present in the official bytecode but is not selected by
`Transformer` and therefore does not affect the current forward graph.

```diff
 def BSRoformer.__init__(
+    linear_transformer_depth=0,
     ...
 ):
```

`linear_transformer_depth` is likewise a compatibility argument recorded by
the official bytecode but unused by this architecture.

The `RMSNorm.forward`, DirectML rotary, MPS/DirectML CPU STFT/mask/ISTFT, and
training-loss device changes are the same as described for the old BS model.
The value-residual transformer flow itself is unchanged from `87576b9`.

### `roformer/mel_band_roformer_new.py`

Changed symbols: `RMSNorm.forward`, `Attention.forward`, and
`MelBandRoformer.forward`.

Compatibility/import changes:

```diff
-from bs_roformer.attend import Attend
+from .attend_new import Attend
+from torch.utils.checkpoint import checkpoint
```

The upstream author's postponed annotations, built-in generics, and PEP 604
unions are retained unchanged.

The local `checkpoint` import matches the official module namespace; the
current restored forward path does not introduce a checkpoint wrapper beyond
what is present in the official bytecode.

The remaining changes mirror the old Mel model:

```diff
+# RMSNorm input follows gamma.device.
+# Rotary q/k temporarily move to CPU on DirectML.
+# Raw audio and FFT reconstruction move to CPU on MPS/DirectML.
+# masks, scatter_indices and denom move to CPU on that path.
+# Returned losses move back to original_device.
```

The anchor's active `LinearAttention`, `linear_transformer_depth`, and
value-residual behavior remain intact.

### `roformer/__init__.py`

Empty UVR package placeholder; no model logic.

## Scope and pyc status

This comparison covers all files borrowed from the five architecture anchors.
Other `lib_v5` modules (`mdxnet.py`, `spec_utils.py`, `tfc_tdf_v3.py`,
`vr_network/*`, Apollo integration, and support modules) were restored from the
official executable but are not derived from the upstream revisions listed
above, so they are outside this upstream-diff table.

Against the official UVR 5.6.1 Python 3.9 pyc files, the restored architecture
logic is semantically aligned except for the explicitly retained Bandit Plus
`torch.jit.script` wrappers noted above. Source formatting, comments, and import
spelling are not expected to be byte-for-byte identical to pyc-derived source.
No checkpoint-container unwrapping or bridge-layer behavior is included in
these architecture files.
