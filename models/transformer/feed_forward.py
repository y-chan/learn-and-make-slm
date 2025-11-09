from torch import nn, Tensor

from models.basic.linear import Linear
from models.transformer.activation import SwiGLU


class FeedForwardSwiGLU(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        d_hidden = d_model * 4

        # 内部的に入力の 4 倍の次元で計算してもとの次元に戻す
        self._swiglu = SwiGLU(d_model, d_hidden)
        self._inverse = Linear(d_hidden, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self._inverse(self._swiglu(x))
