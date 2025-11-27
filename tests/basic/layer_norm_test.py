import torch
from models.basic.layer_norm import LayerNorm


def test_layer_norm_normalizes_last_dimension():
    layer_norm = LayerNorm([10])
    x = torch.randn(4, 10)

    y = layer_norm(x)

    assert y.shape == x.shape
