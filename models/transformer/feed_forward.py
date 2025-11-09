import torch
from models.transformer.activation import SwiGLU


class FeedForwardSwiGLU(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
    ):
        super().__init__()
        self._swiglu = SwiGLU(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._swiglu(x)
