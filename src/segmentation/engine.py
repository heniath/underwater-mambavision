"""Training and evaluation helpers shared by all three experimental conditions."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .metrics import SegmentationMetrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def soft_dice_loss(logits: Tensor, target: Tensor, num_classes: int) -> Tensor:
    probabilities = logits.softmax(dim=1)
    one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).to(probabilities.dtype)
    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
    denominator = (probabilities + one_hot).sum(dim=(0, 2, 3))
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: tuple[str, ...],
) -> dict[str, object]:
    model.eval()
    metrics = SegmentationMetrics(num_classes, class_names)
    loss_total = 0.0
    sample_count = 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        batch_size = images.shape[0]
        loss_total += F.cross_entropy(logits, masks).item() * batch_size
        sample_count += batch_size
        metrics.update(logits, masks)
    result = metrics.compute()
    result["cross_entropy"] = loss_total / max(1, sample_count)
    result["samples"] = sample_count
    return result


def train_condition(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: str | Path,
    class_names: tuple[str, ...],
    epochs: int = 20,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    dice_weight: float = 0.5,
    amp: bool = False,
    initial_state: Optional[dict[str, Tensor]] = None,
) -> dict[str, object]:
    """Train one condition and evaluate its best-validation checkpoint."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    amp_enabled = amp and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):  # PyTorch 2.1 compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    num_classes = len(class_names)
    best_iou = -1.0
    history = []
    checkpoint_path = output / "best_model.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = F.cross_entropy(logits, masks)
                loss = loss + dice_weight * soft_dice_loss(logits, masks, num_classes)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.detach().item() * images.shape[0]
            sample_count += images.shape[0]
        scheduler.step()

        validation = evaluate(model, validation_loader, device, num_classes, class_names)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, sample_count),
            "learning_rate": scheduler.get_last_lr()[0],
            "validation": validation,
        }
        history.append(record)
        print(
            f"epoch {epoch:03d}/{epochs:03d} "
            f"loss={record['train_loss']:.4f} val_mIoU={validation['mean_iou']:.4f}",
            flush=True,
        )
        if float(validation["mean_iou"]) > best_iou:
            best_iou = float(validation["mean_iou"])
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "validation": validation},
                checkpoint_path,
            )
        (output / "history.json").write_text(json.dumps(history, indent=2))

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch 2.1 compatibility
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, test_loader, device, num_classes, class_names)
    result = {
        "best_epoch": checkpoint["epoch"],
        "validation": checkpoint["validation"],
        "test": test_metrics,
        "checkpoint": str(checkpoint_path),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2))
    return result


__all__ = ["evaluate", "seed_everything", "soft_dice_loss", "train_condition"]
