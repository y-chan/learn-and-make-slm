from torch import nn, Tensor

class FeedForwardSwiGLU(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        # TODO: 初期化

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
