from enum import IntEnum
import torch
from models.basic.linear import Linear
from models.transformer.activation import SwiGLU


class FFNHiddenLayerScale(IntEnum):
    PassThrough = 1
    Small = 2
    Base = 4
    Large = 8


class FeedForwardSwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, hidden_scale: FFNHiddenLayerScale = FFNHiddenLayerScale.PassThrough):
        super().__init__()
        d_ff = d_model * hidden_scale

        self._swiglu = SwiGLU(d_model, d_ff)
        self._inverse = Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._inverse(self._swiglu(x))
