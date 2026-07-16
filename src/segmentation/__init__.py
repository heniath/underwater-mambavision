"""Semantic-segmentation components built on the local MambaVision backbone."""

from .data import SUIM_CLASS_NAMES, SUIMDataset, discover_suim_splits, rgb_mask_to_classes
from .model import MambaVisionFPN, build_segmentation_model

__all__ = [
    "MambaVisionFPN",
    "SUIM_CLASS_NAMES",
    "SUIMDataset",
    "build_segmentation_model",
    "discover_suim_splits",
    "rgb_mask_to_classes",
]
