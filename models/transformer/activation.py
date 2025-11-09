import torch

from models.basic.linear import Linear


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + (-x).exp())


class Swish(torch.nn.Module):
    def __init__(self, beta: float = 1.0, mark_beta_learnable: bool = False):
        """
        Implements Swish.

        See: https://arxiv.org/abs/1710.05941

        Parameters
        ----------
        beta : float, optional
            Sharpness parameter, by default 1.0
        mark_beta_learnable : bool, optional
            Whether beta is learnable, by default False

            Why default to False?
            In the original paper, it was shown experimentally that even when beta is marked as a learnable parameter, it converges to a value close to 1.0, indicating that fixing it to 1.0 in practice causes no issues.
        """
        super().__init__()
        if mark_beta_learnable:
            self.beta = torch.nn.Parameter(torch.tensor(beta))
        else:
            self.beta = torch.tensor(beta)

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
