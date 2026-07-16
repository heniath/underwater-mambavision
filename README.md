# Underwater MambaVision segmentation experiment

This repository runs a controlled three-way semantic-segmentation experiment
on [SUIM](https://irvlab.cs.umn.edu/resources/suim-dataset):

1. `raw`: original underwater images with no enhancement.
2. `mambacore`: images restored by UWIR's `hybridmamba_core_3ch` checkpoint.
3. `unet`: images restored by UWIR's `unet_3ch` checkpoint.

All three conditions use the same MambaVision-FPN segmentation architecture,
SUIM masks, train/validation/test membership, initial weights, random seed,
data ordering, augmentation, optimizer, and schedule. Only the input image
condition changes. UWIR is consumed from the separate
[underwater-image-enhancement](https://github.com/heniath/underwater-image-enhancement)
repository; its weights are passed as command-line paths and are not duplicated
here.

The SUIM loader follows the official layout and binary RGB class coding:

```text
SUIM/
├── train_val/
│   ├── images/   # 1,525 images
│   └── masks/
└── TEST/
    ├── images/   # 110 official test images
    └── masks/
```

The segmentation model uses all four MambaVision feature maps in a top-down
FPN decoder and returns `(batch, 8, height, width)` logits. The experiment
reports overall and foreground mIoU, Dice, pixel accuracy, and per-class IoU
and Dice. The default `small` backbone is recommended because the native scan
is educational and sequential; `--backbone t` selects the full 31.8M-parameter
MambaVision-T encoder.

## Kaggle: clone and run

Create a Kaggle GPU notebook and enable Internet. Add two private Kaggle inputs:

- The official SUIM dataset.
- The two UWIR `best_model.pth` files. GitHub cannot host these directly because
  the local checkpoints are larger than its normal per-file limit.

Clone both code repositories:

```bash
%cd /kaggle/working
!git clone https://github.com/heniath/underwater-mambavision.git
!git clone https://github.com/heniath/underwater-image-enhancement.git

!pip install -r /kaggle/working/underwater-mambavision/requirements.txt
!pip install -e /kaggle/working/underwater-image-enhancement
```

Find the attached paths before launching a long run:

```python
from pathlib import Path

for path in Path("/kaggle/input").rglob("best_model.pth"):
    print(path)
for path in Path("/kaggle/input").rglob("train_val"):
    if (path / "images").is_dir() and (path / "masks").is_dir():
        print("SUIM root:", path.parent)
```

Run a one-epoch end-to-end smoke test first:

```bash
%cd /kaggle/working/underwater-mambavision
!python run_experiments.py \
  --data-root /kaggle/input/YOUR-SUIM-PATH/SUIM \
  --uwir-repo /kaggle/working/underwater-image-enhancement \
  --mambacore-checkpoint /kaggle/input/YOUR-WEIGHTS/mambacore_best_model.pth \
  --unet-checkpoint /kaggle/input/YOUR-WEIGHTS/unet_best_model.pth \
  --output-dir /kaggle/working/suim_smoke \
  --epochs 1 --batch-size 4 --workers 2
```

Then run the default 20-epoch experiment:

```bash
!python run_experiments.py \
  --data-root /kaggle/input/YOUR-SUIM-PATH/SUIM \
  --uwir-repo /kaggle/working/underwater-image-enhancement \
  --mambacore-checkpoint /kaggle/input/YOUR-WEIGHTS/mambacore_best_model.pth \
  --unet-checkpoint /kaggle/input/YOUR-WEIGHTS/unet_best_model.pth \
  --output-dir /kaggle/working/suim_three_way \
  --epochs 20 --batch-size 4 --workers 2 --skip-completed
```

`--skip-completed` makes a restarted notebook retain any condition whose
`result.json` is already present. Kaggle's `/kaggle/working` is temporary, so
save a notebook version or export the output after each session. Results are
written to:

```text
suim_three_way/
├── split_manifest.json
├── shared_initial_state.pt
├── enhanced/{mambacore,unet}/...
├── runs/{raw,mambacore,unet}/
│   ├── best_model.pt
│   ├── history.json
│   └── result.json
├── summary.csv
└── summary.json
```

Useful controls:

```bash
# Only generate and cache the two enhanced datasets
python run_experiments.py ... --prepare-only

# Run or resume just one condition
python run_experiments.py ... --conditions raw
python run_experiments.py ... --conditions mambacore --skip-completed

# Full paper-sized encoder (considerably slower, still random initialization)
python run_experiments.py ... --backbone t
```

Mixed precision is off by default because native selective scans are easier to
validate in FP32. `--amp` is available if the selected GPU/model combination is
numerically stable.

## Implementation layout

```text
run_experiments.py              three-condition orchestration
src/segmentation/model.py       MambaVision + FPN model
src/segmentation/data.py        SUIM masks, discovery, split, transforms
src/segmentation/enhancement.py UWIR model/checkpoint adapter
src/segmentation/engine.py      paired training and evaluation
src/segmentation/metrics.py     confusion-matrix metrics
```

## Self-contained MambaVision backbone

This repository contains an educational, CPU-compatible reimplementation of
[**MambaVision: A Hybrid Mamba-Transformer Vision Backbone**](https://openaccess.thecvf.com/content/CVPR2025/papers/Hatamizadeh_MambaVision_A_Hybrid_Mamba-Transformer_Vision_Backbone_CVPR_2025_paper.pdf)
(CVPR 2025). It
reproduces the model's main architectural ideas and selective state-space
recurrence using only PyTorch.  It does not depend on `timm`, `mamba-ssm`,
custom CUDA kernels, pretrained weights, or NVIDIA source code.

The goal is clarity rather than checkpoint or benchmark compatibility with the
[official NVIDIA implementation](https://github.com/NVlabs/MambaVision).

## Architecture

For a 224×224 image, MambaVision-T follows this flow:

| component | representation | spatial size |
|---|---:|---:|
| two-convolution patch stem | 80 channels | 56×56 |
| convolution stage, depth 1 | 80 channels | 56×56 |
| convolution stage, depth 3 | 160 channels | 28×28 |
| window token stage, depth 8 | 320 channels | 14×14 |
| window token stage, depth 4 | 640 channels | 7×7 |
| global average pool + BatchNorm | 640 values | 1×1 |
| linear classifier | 1000 logits | — |

Stages 1 and 2 use residual convolution blocks. Stages 3 and 4 operate on
local, flattened windows. Their first half consists of MambaVision mixers and
their second half consists of multi-head self-attention blocks. Every token
block also contains pre-normalization, an MLP, layer scaling, residual paths,
and stochastic depth. Inputs need not be square or divisible by a window size:
window stages pad on the bottom and right, process the windows, then remove the
padding.

Inside a MambaVision mixer, a linear projection produces two equal-width
branches. Both receive a regular same-padded depthwise 1D convolution and SiLU activation.
One branch drives the selective SSM; the other is a symmetric convolutional
path. Their outputs are concatenated and projected back to the model width.

## Native selective scan

`src/ssm/selective_scan.py` implements the recurrence directly. Given input
`u` and raw step size `delta`, both shaped `(batch, channels, length)`, and a
diagonal continuous state matrix `A` shaped `(channels, state)`, it computes

```text
step_t  = softplus(delta_t + delta_bias)
h_t     = exp(step_t * A) * h_(t-1) + step_t * B_t * u_t
y_t     = sum(C_t * h_t, state) + D * u_t
```

Input-dependent `B` and `C` may be shared across channels with shape
`(batch, state, length)`, as in the mixer, or channel-specific with shape
`(batch, channels, state, length)`. The optional final state has shape
`(batch, channels, state)`. The implementation preserves PyTorch autograd,
device, and input dtype.

Because the scan is a Python loop over the sequence, it is substantially
slower than a fused parallel scan. This is especially noticeable for training
the full T preset on CPU. Small configurations are recommended for experiments
and unit tests.

## Usage

Make `src` importable, then construct the official T-sized preset:

```python
import torch
from mambavision import mambavision_t

model = mambavision_t(num_classes=1000).eval()
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    logits = model(x)                       # (1, 1000)
    embedding = model.forward_embedding(x) # (1, 640)
    maps = model.forward_features(x)        # 56², 28², 14², 7² NCHW maps
```

Feature stages can be selected and reordered:

```python
late_maps = model.forward_features(x, out_indices=(3, 2))
```

For a smaller CPU experiment, provide a configuration directly:

```python
from mambavision import MambaVision, MambaVisionConfig

config = MambaVisionConfig(
    num_classes=10,
    stem_dim=8,
    dims=(16, 24, 32, 48),
    depths=(1, 1, 2, 2),
    num_heads=(1, 2, 4, 6),
    window_sizes=(4, 4, 4, 2),
state_size=4,
    drop_path_rate=0.1,
)
model = MambaVision(config)
```

## Tests

Install the two dependencies and run:

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src pytest
```

The tests compare the scan to an explicit recurrence, check gradients and
window round trips, exercise arbitrary resolutions, verify hybrid block
placement, and validate the MambaVision-T stage dimensions.

## Scope

This project does not include ImageNet training recipes, pretrained weights,
detection or segmentation heads, optimized parallel scans, or benchmark
reproduction. The original scalar SSM learning examples under `src/ssm` remain
independent from the vision model.
