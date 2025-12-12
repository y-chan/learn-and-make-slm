import torch
from models.basic.softmax import Softmax


def test_softmax_outputs_probabilities():
    softmax = Softmax()
    x = torch.randn(2, 8)

    y = softmax(x)

    assert y.shape == x.shape
    sums = y.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
    assert torch.all(y >= 0)


def test_softmax_output():
    softmax = Softmax()
    x = torch.randn(2, 8)

    y = softmax(x)

    y_ref = torch.nn.functional.softmax(x, dim=-1)
    assert y.shape == y_ref.shape
    torch.testing.assert_close(y, y_ref)
