from enum import IntEnum
import torch
from models.basic.linear import Linear
from models.transformer.activation import SwiGLU


class FFNHiddenLayerScale(IntEnum):
    small = 2
    base = 4
    large = 8


class FeedForwardSwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, hidden_scale: FFNHiddenLayerScale = FFNHiddenLayerScale.base):
        super().__init__()
        d_hidden = d_model * hidden_scale

        self._swiglu = SwiGLU(d_model, d_hidden)
        self._inverse = Linear(d_hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._inverse(self._swiglu(x))
