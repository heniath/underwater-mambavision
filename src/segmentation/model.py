"""FPN semantic-segmentation head for the self-contained MambaVision backbone."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mambavision import MambaVision, MambaVisionConfig, mambavision_t


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        # GroupNorm computes statistics even in evaluation. Keep at least two
        # channel values per group so B=H=W=1 remains valid.
        if channels % groups == 0 and channels // groups >= 2:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )


class FPNDecoder(nn.Module):
    """Top-down feature pyramid followed by a multi-scale fusion head."""

    def __init__(self, in_channels: Sequence[int], decoder_dim: int, num_classes: int) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError("FPNDecoder expects four backbone feature widths")
        if num_classes < 2:
            raise ValueError("semantic segmentation requires at least two classes")
        self.laterals = nn.ModuleList(
            nn.Conv2d(channels, decoder_dim, kernel_size=1) for channels in in_channels
        )
        self.refine = nn.ModuleList(
            ConvNormAct(decoder_dim, decoder_dim) for _ in in_channels
        )
        self.fuse = nn.Sequential(
            ConvNormAct(decoder_dim * len(in_channels), decoder_dim),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_dim, num_classes, kernel_size=1),
        )

    def forward(self, features: Sequence[Tensor], output_size: tuple[int, int]) -> Tensor:
        if len(features) != 4:
            raise ValueError("FPNDecoder expects four feature maps")
        pyramid = [projection(feature) for projection, feature in zip(self.laterals, features)]
        for index in range(len(pyramid) - 2, -1, -1):
            pyramid[index] = pyramid[index] + F.interpolate(
                pyramid[index + 1],
                size=pyramid[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        pyramid = [refine(feature) for refine, feature in zip(self.refine, pyramid)]
        finest_size = pyramid[0].shape[-2:]
        fused = torch.cat(
            [
                feature
                if feature.shape[-2:] == finest_size
                else F.interpolate(feature, size=finest_size, mode="bilinear", align_corners=False)
                for feature in pyramid
            ],
            dim=1,
        )
        logits = self.fuse(fused)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class MambaVisionFPN(nn.Module):
    """MambaVision encoder with an FPN semantic-segmentation decoder."""

    def __init__(self, backbone: MambaVision, num_classes: int = 8, decoder_dim: int = 128) -> None:
        super().__init__()
        self.backbone = backbone
        self.decoder = FPNDecoder(backbone.config.dims, decoder_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x: Tensor) -> Tensor:
        output_size = x.shape[-2:]
        features = self.backbone.forward_features(x, out_indices=(0, 1, 2, 3))
        return self.decoder(features, output_size)


def build_segmentation_model(
    variant: str = "small",
    num_classes: int = 8,
    decoder_dim: int = 128,
) -> MambaVisionFPN:
    """Build the practical small model or the full paper-sized T model.

    Both encoders start from random weights; this repository intentionally does
    not download classification checkpoints.
    """
    if variant == "t":
        backbone = mambavision_t(num_classes=0)
    elif variant == "small":
        backbone = MambaVision(
            MambaVisionConfig(
                num_classes=0,
                stem_dim=16,
                dims=(32, 64, 128, 256),
                depths=(1, 2, 4, 2),
                num_heads=(1, 2, 4, 8),
                window_sizes=(8, 8, 8, 4),
                state_size=8,
                drop_path_rate=0.1,
            )
        )
    else:
        raise ValueError(f"unknown backbone variant {variant!r}; choose 'small' or 't'")
    return MambaVisionFPN(backbone, num_classes=num_classes, decoder_dim=decoder_dim)


__all__ = ["FPNDecoder", "MambaVisionFPN", "build_segmentation_model"]
