"""Adapter for applying checkpoints from heniath/underwater-image-enhancement."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn

from .data import SUIMSample


def load_uwir_model(
    uwir_repo: str | Path,
    model_name: str,
    checkpoint_path: str | Path,
    device: torch.device,
) -> nn.Module:
    """Build a UWIR model and load either its training checkpoint or a state dict."""
    repo = Path(uwir_repo).resolve()
    source = repo / "src"
    checkpoint_path = Path(checkpoint_path)
    if not (source / "uwir" / "models").is_dir():
        raise FileNotFoundError(f"not an underwater-image-enhancement checkout: {repo}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"UWIR checkpoint does not exist: {checkpoint_path}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

    from uwir.models import build_model

    model = build_model(model_name, pretrained_backbone=False)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise TypeError(f"unsupported checkpoint payload in {checkpoint_path}")
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)
    return model.eval().to(device)


def _prepare_image(image: Image.Image, max_side: int) -> tuple[Tensor, tuple[int, int], tuple[int, int]]:
    original_size = image.size
    working = image
    if max_side > 0 and max(original_size) > max_side:
        scale = max_side / max(original_size)
        resized = (max(1, round(original_size[0] * scale)), max(1, round(original_size[1] * scale)))
        working = image.resize(resized, Image.Resampling.BILINEAR)
    working_size = working.size
    array = np.asarray(working, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    pad_h = (-tensor.shape[-2]) % 16
    pad_w = (-tensor.shape[-1]) % 16
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
    return tensor, working_size, original_size


@torch.inference_mode()
def enhance_samples(
    model: nn.Module,
    samples: Iterable[SUIMSample],
    output_root: str | Path,
    device: torch.device,
    max_side: int = 512,
    overwrite: bool = False,
) -> int:
    """Enhance unique SUIM images while mirroring their relative keys."""
    output_root = Path(output_root)
    unique = {sample.key: sample for sample in samples}
    written = 0
    for index, (key, sample) in enumerate(sorted(unique.items()), start=1):
        destination = output_root / key
        if destination.is_file() and not overwrite:
            continue
        image = Image.open(sample.image).convert("RGB")
        tensor, working_size, original_size = _prepare_image(image, max_side)
        tensor = tensor.to(device)
        output = model(tensor)
        if not isinstance(output, Tensor) or output.ndim != 4 or output.shape[1] != 3:
            raise TypeError("UWIR model must return an NCHW RGB tensor")
        output = output[0, :, : working_size[1], : working_size[0]].clamp(0, 1)
        array = (output.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
        enhanced = Image.fromarray(array, mode="RGB")
        if enhanced.size != original_size:
            enhanced = enhanced.resize(original_size, Image.Resampling.BILINEAR)
        destination.parent.mkdir(parents=True, exist_ok=True)
        enhanced.save(destination)
        written += 1
        if index == 1 or index % 100 == 0 or index == len(unique):
            print(f"enhanced {index}/{len(unique)} images -> {output_root}", flush=True)
    return written


__all__ = ["enhance_samples", "load_uwir_model"]
