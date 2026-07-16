import torch


def scalar_recurrence(
    x: torch.Tensor,
    a: float,
    b: float,
    c: float,
    d: float,
):
    h = torch.zeros(())
    outputs = []
    states = []

    for x_t in x:
        h = a * h + b * x_t
        y_t = c * h + d * x_t

        states.append(h)
        outputs.append(y_t)

    return torch.stack(outputs), torch.stack(states)

if __name__ == '__main__':
    x = torch.tensor([1.0, 1.2, 0.8, 3.0])

    a = 0.5
    b = 0.3
    c = 0.7
    d = 1.0
    
