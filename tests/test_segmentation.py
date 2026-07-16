from pathlib import Path

import numpy as np
import torch
from PIL import Image

from segmentation.data import SUIM_CLASS_NAMES, discover_suim_splits, rgb_mask_to_classes
from segmentation.enhancement import enhance_samples
from segmentation.metrics import SegmentationMetrics
from segmentation.model import build_segmentation_model


COLORS = np.array(
    [
        (0, 0, 0),
        (0, 0, 255),
        (0, 255, 0),
        (0, 255, 255),
        (255, 0, 0),
        (255, 0, 255),
        (255, 255, 0),
        (255, 255, 255),
    ],
    dtype=np.uint8,
)


def write_pair(root: Path, group: str, index: int) -> None:
    image_dir = root / group / "images"
    mask_dir = root / group / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((9, 13, 3), index, dtype=np.uint8)).save(image_dir / f"{index}.jpg")
    mask = COLORS[np.arange(9 * 13).reshape(9, 13) % 8]
    Image.fromarray(mask).save(mask_dir / f"{index}.png")


def test_suim_color_code_decodes_all_eight_classes():
    decoded = rgb_mask_to_classes(COLORS.reshape(2, 4, 3))
    np.testing.assert_array_equal(decoded.reshape(-1), np.arange(8))


def test_official_suim_folders_and_split_are_discovered(tmp_path):
    for index in range(10):
        write_pair(tmp_path, "train_val", index)
    for index in range(10, 12):
        write_pair(tmp_path, "TEST", index)
    manifest = tmp_path / "manifest.json"
    first = discover_suim_splits(tmp_path, seed=7, validation_fraction=0.2, manifest_path=manifest)
    second = discover_suim_splits(tmp_path, seed=7, validation_fraction=0.2)
    assert {name: len(items) for name, items in first.items()} == {
        "train": 8,
        "validation": 2,
        "test": 2,
    }
    assert first == second
    assert manifest.is_file()


def test_segmentation_model_returns_input_resolution_logits():
    model = build_segmentation_model("small", num_classes=8, decoder_dim=32).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 65, 57))
    assert logits.shape == (2, 8, 65, 57)
    assert torch.isfinite(logits).all()


def test_metrics_are_perfect_for_perfect_logits():
    target = torch.tensor([[[0, 1], [2, 3]]])
    logits = torch.full((1, 8, 2, 2), -10.0)
    logits.scatter_(1, target[:, None], 10.0)
    metrics = SegmentationMetrics(8, SUIM_CLASS_NAMES)
    metrics.update(logits, target)
    result = metrics.compute()
    assert result["pixel_accuracy"] == 1.0
    assert result["mean_iou"] == 1.0
    assert result["mean_dice"] == 1.0


class IdentityEnhancer(torch.nn.Module):
    def forward(self, x):
        return x


def test_enhancement_preserves_keys_and_original_size(tmp_path):
    for index in range(5):
        write_pair(tmp_path, "train_val", index)
    splits = discover_suim_splits(tmp_path, seed=1, validation_fraction=0.2)
    samples = [sample for values in splits.values() for sample in values]
    output = tmp_path / "enhanced"
    enhance_samples(IdentityEnhancer(), samples, output, torch.device("cpu"), max_side=8)
    for sample in samples:
        enhanced = Image.open(output / sample.key)
        assert enhanced.size == Image.open(sample.image).size
