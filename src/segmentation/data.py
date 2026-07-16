"""SUIM discovery, deterministic splitting, and paired image/mask loading."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


SUIM_CLASS_NAMES = (
    "background_waterbody",
    "human_divers",
    "aquatic_plants",
    "wrecks_ruins",
    "robots_instruments",
    "reefs_invertebrates",
    "fish_vertebrates",
    "seafloor_rocks",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


@dataclass(frozen=True)
class SUIMSample:
    image: str
    mask: str
    key: str


def rgb_mask_to_classes(mask: np.ndarray) -> np.ndarray:
    """Decode SUIM's binary RGB color code into class indices 0 through 7."""
    if mask.ndim != 3 or mask.shape[-1] < 3:
        raise ValueError(f"expected an RGB mask, got shape {mask.shape}")
    bits = mask[..., :3] >= 128
    return (bits[..., 0] * 4 + bits[..., 1] * 2 + bits[..., 2]).astype(np.int64)


def _files_by_stem(directory: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def _pair_folder(data_root: Path, image_dir: Path) -> list[SUIMSample]:
    mask_dir = image_dir.parent / "masks"
    if not mask_dir.is_dir():
        return []
    images = _files_by_stem(image_dir)
    masks = _files_by_stem(mask_dir)
    samples = []
    for stem in sorted(images.keys() & masks.keys()):
        image_path = images[stem]
        samples.append(
            SUIMSample(
                image=str(image_path),
                mask=str(masks[stem]),
                key=str(image_path.relative_to(data_root).with_suffix(".png")),
            )
        )
    return samples


def _find_named_pair(data_root: Path, parent_name: str) -> list[SUIMSample]:
    candidates = []
    for image_dir in data_root.rglob("images"):
        if image_dir.parent.name.lower() == parent_name.lower():
            paired = _pair_folder(data_root, image_dir)
            if paired:
                candidates.append(paired)
    return max(candidates, key=len, default=[])


def discover_suim_splits(
    data_root: str | Path,
    seed: int = 42,
    validation_fraction: float = 0.15,
    manifest_path: Optional[str | Path] = None,
) -> dict[str, list[SUIMSample]]:
    """Use SUIM's official test folder and split train_val deterministically.

    If the official folder names are absent, the largest paired ``images`` / 
    ``masks`` directory is deterministically divided into train/validation/test.
    """
    root = Path(data_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"SUIM data root does not exist: {root}")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")

    train_val = _find_named_pair(root, "train_val")
    official_test = _find_named_pair(root, "TEST")
    rng = random.Random(seed)

    if train_val:
        shuffled = train_val.copy()
        rng.shuffle(shuffled)
        validation_count = max(1, round(len(shuffled) * validation_fraction))
        splits = {
            "train": shuffled[validation_count:],
            "validation": shuffled[:validation_count],
            "test": official_test,
        }
        if not official_test:
            test_count = validation_count
            splits["test"] = splits["train"][:test_count]
            splits["train"] = splits["train"][test_count:]
    else:
        candidates = [_pair_folder(root, path) for path in root.rglob("images")]
        samples = max((items for items in candidates if items), key=len, default=[])
        if len(samples) < 3:
            raise FileNotFoundError(
                f"could not find paired SUIM images/masks beneath {root}; "
                "expected train_val/images and train_val/masks"
            )
        rng.shuffle(samples)
        holdout = max(1, round(len(samples) * validation_fraction))
        splits = {
            "train": samples[holdout * 2 :],
            "validation": samples[:holdout],
            "test": samples[holdout : holdout * 2],
        }

    if not all(splits.values()):
        raise ValueError(f"one or more dataset splits are empty: { {k: len(v) for k, v in splits.items()} }")
    if manifest_path is not None:
        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({name: [asdict(sample) for sample in items] for name, items in splits.items()}, indent=2)
        )
    return splits


class SUIMDataset(Dataset[tuple[Tensor, Tensor]]):
    """Load raw or pre-enhanced images with the unchanged SUIM masks."""

    def __init__(
        self,
        samples: list[SUIMSample],
        image_size: int = 256,
        enhanced_root: Optional[str | Path] = None,
        augment: bool = False,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.enhanced_root = Path(enhanced_root) if enhanced_root is not None else None
        self.augment = augment
        if image_size <= 0:
            raise ValueError("image_size must be positive")

    def __len__(self) -> int:
        return len(self.samples)

    def _image_path(self, sample: SUIMSample) -> Path:
        path = Path(sample.image) if self.enhanced_root is None else self.enhanced_root / sample.key
        if not path.is_file():
            raise FileNotFoundError(f"condition image is missing: {path}")
        return path

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sample = self.samples[index]
        size = (self.image_size, self.image_size)
        image = Image.open(self._image_path(sample)).convert("RGB").resize(
            size, Image.Resampling.BILINEAR
        )
        mask = Image.open(sample.mask).convert("RGB").resize(size, Image.Resampling.NEAREST)
        image_array = np.asarray(image, dtype=np.float32).copy() / 255.0
        mask_array = rgb_mask_to_classes(np.asarray(mask).copy())
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_array)
        if self.augment and torch.rand(()) < 0.5:
            image_tensor = image_tensor.flip(-1)
            mask_tensor = mask_tensor.flip(-1)
        image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD
        return image_tensor, mask_tensor


__all__ = [
    "SUIM_CLASS_NAMES",
    "SUIMDataset",
    "SUIMSample",
    "discover_suim_splits",
    "rgb_mask_to_classes",
]
