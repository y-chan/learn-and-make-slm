import torch

from models.basic.linear import Linear
from models.transformer.activation import SwiGLU, Swish


class FeedForwardSwiGLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
    ):
        super().__init__()
        self._swiglu = SwiGLU(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._swiglu(x)


class FeedForwardSwish(torch.nn.Module):
    """
    GPT-2の再現であればGELUを使うべきだが、
    めんどくさいのでだいたい同じのSwishを使う
    """
    def __init__(
        self,
        d_model: int,
    ):
        super().__init__()
        self.swish = Swish()
        self.linear_in = Linear(d_model, d_model)
        self.linear_out = Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        x = self.swish(x)
        x = self.linear_out(x)
        return x
