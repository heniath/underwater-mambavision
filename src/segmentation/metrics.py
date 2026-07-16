"""Streaming confusion-matrix metrics for semantic segmentation."""

from __future__ import annotations

import torch
from torch import Tensor


class SegmentationMetrics:
    def __init__(self, num_classes: int, class_names: tuple[str, ...]) -> None:
        if len(class_names) != num_classes:
            raise ValueError("class_names length must match num_classes")
        self.num_classes = num_classes
        self.class_names = class_names
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    def update(self, logits: Tensor, target: Tensor) -> None:
        prediction = logits.argmax(dim=1).detach().cpu().reshape(-1)
        target = target.detach().cpu().reshape(-1)
        valid = (target >= 0) & (target < self.num_classes)
        encoded = target[valid] * self.num_classes + prediction[valid]
        self.confusion += torch.bincount(
            encoded, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, object]:
        matrix = self.confusion.to(torch.float64)
        true_positive = matrix.diag()
        target_count = matrix.sum(dim=1)
        predicted_count = matrix.sum(dim=0)
        union = target_count + predicted_count - true_positive
        denominator = target_count + predicted_count
        iou = torch.where(union > 0, true_positive / union, torch.nan)
        dice = torch.where(denominator > 0, 2 * true_positive / denominator, torch.nan)
        total = matrix.sum()
        pixel_accuracy = (true_positive.sum() / total).item() if total else float("nan")
        foreground_iou = iou[1:]
        return {
            "pixel_accuracy": pixel_accuracy,
            "mean_iou": torch.nanmean(iou).item(),
            "foreground_mean_iou": torch.nanmean(foreground_iou).item(),
            "mean_dice": torch.nanmean(dice).item(),
            "per_class_iou": {
                name: None if torch.isnan(value) else value.item()
                for name, value in zip(self.class_names, iou)
            },
            "per_class_dice": {
                name: None if torch.isnan(value) else value.item()
                for name, value in zip(self.class_names, dice)
            },
        }


__all__ = ["SegmentationMetrics"]
