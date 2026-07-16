"""Readable, native-PyTorch reimplementation of the MambaVision-T backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ssm.selective_scan import selective_scan


def window_partition(x: Tensor, window_size: int) -> Tuple[Tensor, Tuple[int, int], Tuple[int, int]]:
    """Pad and partition an NCHW map into flattened channel-last windows.

    Returns ``(windows, original_hw, padded_hw)``.  ``windows`` has shape
    ``(batch * number_of_windows, window_size**2, channels)``.
    """
    if x.ndim != 4:
        raise ValueError(f"expected an NCHW tensor, got shape {tuple(x.shape)}")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    batch, channels, height, width = x.shape
    pad_h = (-height) % window_size
    pad_w = (-width) % window_size
    x = F.pad(x, (0, pad_w, 0, pad_h))
    padded_h, padded_w = height + pad_h, width + pad_w
    x = x.reshape(
        batch,
        channels,
        padded_h // window_size,
        window_size,
        padded_w // window_size,
        window_size,
    )
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, channels)
    return windows, (height, width), (padded_h, padded_w)


def window_reverse(
    windows: Tensor,
    window_size: int,
    original_hw: Tuple[int, int],
    padded_hw: Tuple[int, int],
) -> Tensor:
    """Reverse :func:`window_partition` and remove its right/bottom padding."""
    height, width = original_hw
    padded_h, padded_w = padded_hw
    windows_per_image = (padded_h // window_size) * (padded_w // window_size)
    if windows.ndim != 3 or windows.shape[1] != window_size * window_size:
        raise ValueError("windows have an incompatible shape")
    if windows_per_image == 0 or windows.shape[0] % windows_per_image:
        raise ValueError("window count is incompatible with the padded spatial size")
    batch = windows.shape[0] // windows_per_image
    channels = windows.shape[-1]
    x = windows.reshape(
        batch,
        padded_h // window_size,
        padded_w // window_size,
        window_size,
        window_size,
        channels,
    )
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(batch, channels, padded_h, padded_w)
    return x[:, :, :height, :width].contiguous()


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, x: Tensor) -> Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class PatchStem(nn.Module):
    def __init__(self, in_channels: int, stem_dim: int, out_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, stem_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),
            nn.Conv2d(stem_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class Downsample(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class ConvBlock(nn.Module):
    """Residual convolutional block used in the two high-resolution stages."""

    def __init__(self, dim: int, drop_path: float, layer_scale: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        self.act = nn.GELU()
        self.drop_path = DropPath(drop_path)
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim)) if layer_scale > 0 else None

    def forward(self, x: Tensor) -> Tensor:
        residual = self.act(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))
        if self.gamma is not None:
            residual = residual * self.gamma[None, :, None, None]
        return x + self.drop_path(residual)


class MLP(nn.Module):
    def __init__(self, dim: int, ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(F.gelu(self.fc1(x)))
        return self.dropout(self.fc2(x))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dimension {dim} must be divisible by {num_heads} heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, channels = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = self.attn_drop(attention.softmax(dim=-1))
        x = (attention @ v).transpose(1, 2).reshape(batch, length, channels)
        return self.proj_drop(self.proj(x))


class MambaVisionMixer(nn.Module):
    """Selective SSM path paired with an equal-width gated convolution path."""

    def __init__(
        self,
        dim: int,
        state_size: int = 16,
        expand: int = 1,
        conv_kernel: int = 3,
        dt_rank: Optional[int] = None,
    ) -> None:
        super().__init__()
        inner_dim = dim * expand
        if inner_dim % 2:
            raise ValueError("expanded mixer dimension must be even")
        self.branch_dim = inner_dim // 2
        self.state_size = state_size
        self.dt_rank = dt_rank or max(1, (dim + 15) // 16)
        self.conv_kernel = conv_kernel

        self.in_proj = nn.Linear(dim, inner_dim)
        self.conv_x = nn.Conv1d(
            self.branch_dim, self.branch_dim, conv_kernel, groups=self.branch_dim,
            padding="same",
        )
        self.conv_z = nn.Conv1d(
            self.branch_dim, self.branch_dim, conv_kernel, groups=self.branch_dim,
            padding="same",
        )
        self.x_proj = nn.Linear(self.branch_dim, self.dt_rank + state_size * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.branch_dim)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, state_size + 1, dtype=torch.float32))[None, :]
            .repeat(self.branch_dim, 1)
        )
        self.D = nn.Parameter(torch.ones(self.branch_dim))
        self.out_proj = nn.Linear(inner_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        length = x.shape[1]
        projected = self.in_proj(x).transpose(1, 2)
        x_path, z_path = projected.chunk(2, dim=1)
        # Unlike language Mamba, vision tokens have no useful causal ordering.
        # A regular same-padded convolution sees both adjacent directions.
        x_path = F.silu(self.conv_x(x_path))
        z_path = F.silu(self.conv_z(z_path))

        parameters = self.x_proj(x_path.transpose(1, 2))
        dt_low_rank, B, C = torch.split(
            parameters, [self.dt_rank, self.state_size, self.state_size], dim=-1
        )
        delta = F.linear(dt_low_rank, self.dt_proj.weight).transpose(1, 2)
        B = B.transpose(1, 2)
        C = C.transpose(1, 2)
        A = -torch.exp(self.A_log)
        y = selective_scan(
            x_path,
            delta,
            A,
            B,
            C,
            D=self.D,
            delta_bias=self.dt_proj.bias,
        )
        if y.shape[-1] != length:  # defensive assertion around convolution trimming
            raise RuntimeError("mixer changed the token sequence length")
        return self.out_proj(torch.cat((y, z_path), dim=1).transpose(1, 2))


class TokenBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        mixer: nn.Module,
        mlp_ratio: float,
        dropout: float,
        drop_path: float,
        layer_scale: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = mixer
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)
        self.drop_path = DropPath(drop_path)
        self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim)) if layer_scale > 0 else None
        self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim)) if layer_scale > 0 else None

    def _scale(self, x: Tensor, gamma: Optional[nn.Parameter]) -> Tensor:
        return x if gamma is None else x * gamma

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self._scale(self.mixer(self.norm1(x)), self.gamma1))
        return x + self.drop_path(self._scale(self.mlp(self.norm2(x)), self.gamma2))


class WindowStage(nn.Module):
    def __init__(self, blocks: Sequence[TokenBlock], window_size: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.window_size = window_size

    def forward(self, x: Tensor) -> Tensor:
        windows, original_hw, padded_hw = window_partition(x, self.window_size)
        for block in self.blocks:
            windows = block(windows)
        return window_reverse(windows, self.window_size, original_hw, padded_hw)


@dataclass(frozen=True)
class MambaVisionConfig:
    """Configuration for a four-stage MambaVision model."""

    in_channels: int = 3
    num_classes: int = 1000
    stem_dim: int = 32
    dims: Tuple[int, int, int, int] = (80, 160, 320, 640)
    depths: Tuple[int, int, int, int] = (1, 3, 8, 4)
    num_heads: Tuple[int, int, int, int] = (2, 4, 8, 16)
    window_sizes: Tuple[int, int, int, int] = (8, 8, 14, 7)
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    drop_path_rate: float = 0.2
    layer_scale_init_value: float = 1e-5
    state_size: int = 16
    mixer_expand: int = 1
    conv_kernel: int = 3
    out_indices: Tuple[int, ...] = (0, 1, 2, 3)

    def __post_init__(self) -> None:
        for name in ("dims", "depths", "num_heads", "window_sizes"):
            if len(getattr(self, name)) != 4:
                raise ValueError(f"{name} must contain four stage values")
        if any(depth <= 0 for depth in self.depths):
            raise ValueError("all stage depths must be positive")
        if not 0.0 <= self.drop_path_rate < 1.0:
            raise ValueError("drop_path_rate must be in [0, 1)")
        if any(index not in range(4) for index in self.out_indices):
            raise ValueError("out_indices must only contain values from 0 to 3")


class MambaVision(nn.Module):
    def __init__(self, config: MambaVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.stem = PatchStem(config.in_channels, config.stem_dim, config.dims[0])
        self.downsamples = nn.ModuleList(
            [Downsample(config.dims[i], config.dims[i + 1]) for i in range(3)]
        )

        total_blocks = sum(config.depths)
        rates = torch.linspace(0, config.drop_path_rate, total_blocks).tolist()
        rate_index = 0
        stages = []
        for stage_index, (dim, depth) in enumerate(zip(config.dims, config.depths)):
            if stage_index < 2:
                blocks = []
                for _ in range(depth):
                    blocks.append(
                        ConvBlock(dim, rates[rate_index], config.layer_scale_init_value)
                    )
                    rate_index += 1
                stages.append(nn.Sequential(*blocks))
                continue

            token_blocks = []
            for block_index in range(depth):
                # The Mamba half precedes the attention half, as in MambaVision.
                if block_index < depth // 2:
                    mixer: nn.Module = MambaVisionMixer(
                        dim,
                        state_size=config.state_size,
                        expand=config.mixer_expand,
                        conv_kernel=config.conv_kernel,
                    )
                else:
                    mixer = Attention(dim, config.num_heads[stage_index], config.dropout)
                token_blocks.append(
                    TokenBlock(
                        dim,
                        mixer,
                        config.mlp_ratio,
                        config.dropout,
                        rates[rate_index],
                        config.layer_scale_init_value,
                    )
                )
                rate_index += 1
            stages.append(WindowStage(token_blocks, config.window_sizes[stage_index]))
        self.stages = nn.ModuleList(stages)
        self.head_norm = nn.BatchNorm1d(config.dims[-1])
        self.head = (
            nn.Linear(config.dims[-1], config.num_classes)
            if config.num_classes > 0
            else nn.Identity()
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(
        self, x: Tensor, out_indices: Optional[Sequence[int]] = None
    ) -> Tuple[Tensor, ...]:
        """Return selected stage maps as an ordered tuple of NCHW tensors."""
        indices = tuple(self.config.out_indices if out_indices is None else out_indices)
        if any(index not in range(4) for index in indices):
            raise ValueError("out_indices must only contain values from 0 to 3")
        if len(set(indices)) != len(indices):
            raise ValueError("out_indices must not contain duplicates")
        requested = set(indices)
        features = {}
        x = self.stem(x)
        for stage_index, stage in enumerate(self.stages):
            if stage_index:
                x = self.downsamples[stage_index - 1](x)
            x = stage(x)
            if stage_index in requested:
                features[stage_index] = x
        return tuple(features[index] for index in indices)

    def _final_map(self, x: Tensor) -> Tensor:
        # Avoid retaining all stage outputs when only classification is needed.
        x = self.stem(x)
        for stage_index, stage in enumerate(self.stages):
            if stage_index:
                x = self.downsamples[stage_index - 1](x)
            x = stage(x)
        return x

    def forward_embedding(self, x: Tensor) -> Tensor:
        """Return the batch-normalized, globally pooled final representation."""
        x = self._final_map(x).mean(dim=(-2, -1))
        return self.head_norm(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.forward_embedding(x))


def mambavision_t(**kwargs: object) -> MambaVision:
    """Construct MambaVision-T, optionally overriding configuration fields."""
    return MambaVision(MambaVisionConfig(**kwargs))


__all__ = [
    "Attention",
    "DropPath",
    "MambaVision",
    "MambaVisionConfig",
    "MambaVisionMixer",
    "mambavision_t",
    "window_partition",
    "window_reverse",
]
