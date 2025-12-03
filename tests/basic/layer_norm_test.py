import torch
from models.basic.layer_norm import LayerNorm


def test_layer_norm_normalizes_last_dimension():
    layer_norm = LayerNorm([10])
    x = torch.randn(4, 10)

    y = layer_norm(x)

    assert y.shape == x.shape


def test_layer_norm_output():
    layer_norm = LayerNorm([10])
    x = torch.randn(4, 10)

    y = layer_norm(x)

    y_ref = torch.nn.functional.layer_norm(
        x, layer_norm.normalized_shape, layer_norm.gamma, layer_norm.beta, eps=layer_norm.eps
    )
    assert y.shape == y_ref.shape
    torch.testing.assert_close(y, y_ref)
