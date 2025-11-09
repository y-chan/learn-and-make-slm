import torch

from models.basic.linear import Linear


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + (-x).exp())


class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * sigmoid(self.beta * x)


class GLU(torch.nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
    ):
        super().__init__()
        self.W = Linear(dim_in, dim_hidden)
        self.V = Linear(dim_in, dim_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W(x) * sigmoid(self.V(x))


class SwiGLU(torch.nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
    ):
        super().__init__()
        self.W = Linear(dim_in, dim_hidden)
        self.V = Linear(dim_in, dim_hidden)
        self._swish = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W(x) * self._swish(self.V(x))
