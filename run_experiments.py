"""Run the controlled raw/MambaCore/U-Net SUIM segmentation experiment."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from segmentation.data import SUIM_CLASS_NAMES, SUIMDataset, discover_suim_splits
from segmentation.engine import seed_everything, train_condition
from segmentation.enhancement import enhance_samples, load_uwir_model
from segmentation.model import build_segmentation_model


CONDITIONS = ("raw", "mambacore", "unet")
UWIR_MODELS = {
    "mambacore": "hybridmamba_core_3ch",
    "unet": "unet_3ch",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SUIM segmentation on raw, MambaCore-enhanced, and U-Net-enhanced images."
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing SUIM train_val and TEST")
    parser.add_argument(
        "--uwir-repo",
        type=Path,
        default=Path("../underwater-image-enhancement"),
        help="Checkout of heniath/underwater-image-enhancement",
    )
    parser.add_argument("--mambacore-checkpoint", type=Path)
    parser.add_argument("--unet-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/suim_three_way"))
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--backbone", choices=("small", "t"), default="small")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--decoder-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--enhancement-max-side", type=int, default=512)
    parser.add_argument("--overwrite-enhanced", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA mixed precision (off by default for native scans)")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or an explicit CUDA device")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("a CUDA device was requested but CUDA is unavailable")
    return device


def make_loaders(splits, image_size, enhanced_root, batch_size, workers, seed):
    datasets = {
        name: SUIMDataset(
            samples,
            image_size=image_size,
            enhanced_root=enhanced_root,
            augment=name == "train",
        )
        for name, samples in splits.items()
    }
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    return {
        # BatchNorm2d at a 1x1 final map cannot estimate variance from a
        # singleton batch, so omit only an incomplete singleton train batch.
        "train": DataLoader(
            datasets["train"],
            shuffle=True,
            generator=generator,
            drop_last=len(datasets["train"]) % batch_size == 1,
            **common,
        ),
        "validation": DataLoader(datasets["validation"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }


def prepare_enhancements(args, splits, device):
    all_samples = [sample for samples in splits.values() for sample in samples]
    roots = {}
    for condition in args.conditions:
        if condition == "raw":
            roots[condition] = None
            continue
        checkpoint = getattr(args, f"{condition}_checkpoint")
        if checkpoint is None:
            raise ValueError(f"--{condition}-checkpoint is required for the {condition} condition")
        destination = args.output_dir / "enhanced" / condition
        print(f"\nPreparing {condition} images with {UWIR_MODELS[condition]}")
        enhancer = load_uwir_model(args.uwir_repo, UWIR_MODELS[condition], checkpoint, device)
        enhance_samples(
            enhancer,
            all_samples,
            destination,
            device,
            max_side=args.enhancement_max_side,
            overwrite=args.overwrite_enhanced,
        )
        del enhancer
        if device.type == "cuda":
            torch.cuda.empty_cache()
        roots[condition] = destination
    return roots


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.batch_size < 2 and not args.prepare_only:
        raise ValueError("training batch size must be at least 2 because the backbone uses BatchNorm")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"device={device} conditions={args.conditions}")
    splits = discover_suim_splits(
        args.data_root,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        manifest_path=args.output_dir / "split_manifest.json",
    )
    print("split sizes:", {name: len(samples) for name, samples in splits.items()})
    condition_roots = prepare_enhancements(args, splits, device)
    if args.prepare_only:
        print("Enhancement preparation complete; training skipped.")
        return

    seed_everything(args.seed)
    reference_model = build_segmentation_model(
        args.backbone, num_classes=len(SUIM_CLASS_NAMES), decoder_dim=args.decoder_dim
    )
    initial_state = copy.deepcopy(reference_model.state_dict())
    torch.save(initial_state, args.output_dir / "shared_initial_state.pt")
    del reference_model

    results = {}
    for condition in args.conditions:
        condition_dir = args.output_dir / "runs" / condition
        result_path = condition_dir / "result.json"
        if args.skip_completed and result_path.is_file():
            print(f"Skipping completed condition: {condition}")
            results[condition] = json.loads(result_path.read_text())
            continue
        print(f"\n{'=' * 72}\nTraining condition: {condition}\n{'=' * 72}")
        # Reset every source of randomness so augmentation, drop path, and data
        # ordering are paired across conditions as closely as possible.
        seed_everything(args.seed)
        loaders = make_loaders(
            splits,
            args.image_size,
            condition_roots[condition],
            args.batch_size,
            args.workers,
            args.seed,
        )
        model = build_segmentation_model(
            args.backbone, num_classes=len(SUIM_CLASS_NAMES), decoder_dim=args.decoder_dim
        )
        results[condition] = train_condition(
            model,
            loaders["train"],
            loaders["validation"],
            loaders["test"],
            device,
            condition_dir,
            SUIM_CLASS_NAMES,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            dice_weight=args.dice_weight,
            amp=args.amp,
            initial_state=initial_state,
        )
        del model, loaders
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("condition", "best_epoch", "mean_iou", "foreground_mean_iou", "mean_dice", "pixel_accuracy"),
        )
        writer.writeheader()
        for condition, result in results.items():
            test = result["test"]
            writer.writerow(
                {
                    "condition": condition,
                    "best_epoch": result["best_epoch"],
                    "mean_iou": test["mean_iou"],
                    "foreground_mean_iou": test["foreground_mean_iou"],
                    "mean_dice": test["mean_dice"],
                    "pixel_accuracy": test["pixel_accuracy"],
                }
            )
    print(f"\nExperiment complete. Summary: {summary_path}")


if __name__ == "__main__":
    main()
