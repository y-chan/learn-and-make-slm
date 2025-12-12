import torch
from models.basic.linear import Linear


def test_linear_output_shape_and_dtype():
    linear = Linear(6, 3)
    x = torch.randn(5, 6)

    y = linear(x)

    assert y.shape == (5, 3)
    assert y.dtype == x.dtype


def test_linear_output():
    linear = Linear(6, 3)
    x = torch.randn(5, 6)

    y = linear(x)

    y_ref = torch.nn.functional.linear(x, linear.weight, linear.bias)
    assert y.shape == y_ref.shape
    torch.testing.assert_close(y, y_ref)
