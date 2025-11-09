from torch import nn, Tensor

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        # TODO: 初期化

    def forward(self, Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
        raise NotImplementedError


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        # TODO: 初期化

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


