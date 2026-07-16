import torch
import torch.nn.functional as F

from ssm.selective_scan import selective_scan


def explicit_scan(u, delta, A, B, C, D, bias):
    steps = F.softplus(delta + bias[None, :, None])
    state = u.new_zeros(u.shape[0], u.shape[1], A.shape[1])
    outputs = []
    for t in range(u.shape[-1]):
        step = steps[:, :, t, None]
        state = torch.exp(step * A[None]) * state
        state = state + step * B[:, None, :, t] * u[:, :, t, None]
        outputs.append((state * C[:, None, :, t]).sum(-1) + D * u[:, :, t])
    return torch.stack(outputs, -1), state


def test_selective_scan_matches_explicit_recurrence_and_final_state():
    torch.manual_seed(3)
    u = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    delta = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    A = -torch.rand(3, 4, dtype=torch.float64, requires_grad=True)
    B = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True)
    C = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True)
    D = torch.randn(3, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(3, dtype=torch.float64, requires_grad=True)

    actual_y, actual_state = selective_scan(
        u, delta, A, B, C, D, bias, return_last_state=True
    )
    expected_y, expected_state = explicit_scan(u, delta, A, B, C, D, bias)
    torch.testing.assert_close(actual_y, expected_y)
    torch.testing.assert_close(actual_state, expected_state)

    loss = actual_y.square().mean() + actual_state.square().mean()
    gradients = torch.autograd.grad(loss, (u, delta, A, B, C, D, bias))
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)


def test_channel_specific_b_and_c_are_supported():
    u = torch.ones(1, 2, 3)
    delta = torch.zeros_like(u)
    A = -torch.ones(2, 2)
    B = torch.randn(1, 2, 2, 3)
    C = torch.randn(1, 2, 2, 3)
    output = selective_scan(u, delta, A, B, C)
    assert output.shape == u.shape
    assert torch.isfinite(output).all()
