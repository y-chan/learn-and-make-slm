import torch

from models.transformer.rope import RotaryPositionalEncoding


def test_rope():
    x = torch.randn(10, 16)  # (seq_len, dim)
    rope = RotaryPositionalEncoding(dim=16)
    y = rope(x)

    assert y.shape == x.shape
