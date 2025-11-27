from torch import nn, Tensor


class Decoder(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int):
        super().__init__()
        # TODO: 初期化

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
