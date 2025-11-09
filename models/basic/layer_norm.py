from torch import nn, Tensor

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int | list[int]):
        super().__init__()
        # TODO: 初期化

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


if __name__ == "__main__":
    import torch

    layer_norm = LayerNorm([10])

    x = torch.randn(2, 10)
    assert layer_norm(x).shape == (2, 10)
    print("LayerNorm test passed")
