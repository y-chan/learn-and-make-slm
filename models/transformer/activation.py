import torch


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + (-x).exp())


class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * sigmoid(self.beta * x)
