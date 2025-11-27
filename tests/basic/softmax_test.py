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
