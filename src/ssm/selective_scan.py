
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor


def _prepare_bc(value: Tensor, batch: int, channels: int, state_size: int, length: int, name: str) -> Tensor:
    """Turn a shared ``(B, N, L)`` B/C tensor into ``(B, D, N, L)``."""
    if value.ndim == 3:
        if value.shape != (batch, state_size, length):
            raise ValueError(
                f"{name} must have shape (batch, state, length); got {tuple(value.shape)}"
            )
        return value[:, None, :, :].expand(-1, channels, -1, -1)
    if value.ndim == 4:
        if value.shape != (batch, channels, state_size, length):
            raise ValueError(
                f"{name} must have shape (batch, channels, state, length); "
                f"got {tuple(value.shape)}"
            )
        return value
    raise ValueError(f"{name} must be a 3D or 4D tensor, got {value.ndim} dimensions")


def selective_scan(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Optional[Tensor] = None,
    delta_bias: Optional[Tensor] = None,
    *,
    return_last_state: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    r"""Run the input-dependent selective SSM recurrence.

    Args:
        u: Input with shape ``(batch, channels, length)``.
        delta: Per-token step sizes with the same shape as ``u``.
        A: Continuous state matrix diagonal, shape ``(channels, state)``.
        B, C: Input-dependent projections.  They can be shared across channels
            with shape ``(batch, state, length)`` or be channel-specific with
            shape ``(batch, channels, state, length)``.
        D: Optional skip weights, shape ``(channels,)``.
        delta_bias: Optional bias added before softplus, shape ``(channels,)``.
        return_last_state: Also return the final ``(batch, channels, state)``
            recurrent state.

    The discretized recurrence is

    .. math::

       \Delta_t &= \operatorname{softplus}(\delta_t + b_\Delta) \\
       h_t &= \exp(\Delta_t A) h_{t-1} + \Delta_t B_t u_t \\
       y_t &= C_t h_t + D u_t.

    All operations are native PyTorch operations, so device, dtype and
    automatic differentiation follow the input tensors.
    """
    if u.ndim != 3:
        raise ValueError(f"u must have shape (batch, channels, length), got {tuple(u.shape)}")
    if delta.shape != u.shape:
        raise ValueError(f"delta must have the same shape as u; got {tuple(delta.shape)}")
    if A.ndim != 2 or A.shape[0] != u.shape[1]:
        raise ValueError(
            f"A must have shape (channels, state), with channels={u.shape[1]}; "
            f"got {tuple(A.shape)}"
        )

    batch, channels, length = u.shape
    state_size = A.shape[1]
    B_full = _prepare_bc(B, batch, channels, state_size, length, "B")
    C_full = _prepare_bc(C, batch, channels, state_size, length, "C")

    if D is not None and D.shape != (channels,):
        raise ValueError(f"D must have shape ({channels},), got {tuple(D.shape)}")
    if delta_bias is not None and delta_bias.shape != (channels,):
        raise ValueError(
            f"delta_bias must have shape ({channels},), got {tuple(delta_bias.shape)}"
        )

    # Parameters are normally already colocated by nn.Module.  Explicit casts
    # make the standalone function predictable without breaking gradient flow.
    A_work = A.to(device=u.device, dtype=u.dtype)
    B_full = B_full.to(device=u.device, dtype=u.dtype)
    C_full = C_full.to(device=u.device, dtype=u.dtype)
    raw_delta = delta.to(device=u.device, dtype=u.dtype)
    if delta_bias is not None:
        raw_delta = raw_delta + delta_bias.to(device=u.device, dtype=u.dtype)[None, :, None]
    steps = F.softplus(raw_delta)

    state = u.new_zeros((batch, channels, state_size))
    outputs = []
    for index in range(length):
        step = steps[:, :, index, None]
        input_t = u[:, :, index, None]
        state = torch.exp(step * A_work[None, :, :]) * state
        state = state + step * B_full[:, :, :, index] * input_t
        output_t = (state * C_full[:, :, :, index]).sum(dim=-1)
        if D is not None:
            output_t = output_t + D.to(device=u.device, dtype=u.dtype)[None, :] * u[:, :, index]
        outputs.append(output_t)

    # Stacking an empty list is undefined. Vision sequences are never empty,
    # but a clear validation error is friendlier for standalone use.
    if not outputs:
        raise ValueError("the sequence length must be greater than zero")
    output = torch.stack(outputs, dim=-1)
    return (output, state) if return_last_state else output


__all__ = ["selective_scan"]
