import torch

def discretize_scalar_ssm(
    a: torch.Tensor,
    b: torch.Tensor,
    delta: torch.Tensor,
    eps: float = 1e-8,
):
    a_bar = torch.exp(delta * a)

    if torch.abs(a) < eps:
        b_bar = delta * b
    else:
        b_bar = ((a_bar - 1.0) / a) * b

    return a_bar, b_bar
