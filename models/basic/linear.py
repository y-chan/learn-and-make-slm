from torch import nn, Tensor

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # TODO: 初期化

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


