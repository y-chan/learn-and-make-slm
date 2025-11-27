from torch import Tensor, nn
import torch


def rotate_half(x: Tensor) -> Tensor:
    # x: [..., dim]
    x = x.reshape(*x.shape[:-1], -1, 2)  # [..., dim/2, 2]
    x1, x2 = x[..., 0], x[..., 1]
    rotated = torch.stack((-x2, x1), dim=-1)  # [..., dim/2, 2]
    return rotated.reshape(*x.shape[:-2], -1)  # [..., dim]


class RotaryPositionalEncoding(nn.Module):
    thetas: Tensor  # (dim/2,)

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        _thetas = self._setup_thetas(dim)
        self.register_buffer("thetas", _thetas)

    def _setup_thetas(self, dim: int) -> Tensor:
        base = 10000
        vec = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))  # (dim/2,)
        return torch.repeat_interleave(vec, repeats=2)  # (dim,)

    def forward(self, x: Tensor) -> Tensor:
        # assert x dim is same as self.dim
        assert x.size(-1) == self.dim, f"x dim {x.size(-1)} must be same as rope dim {self.dim}"

        pos = torch.arange(1, 1 + x.size(0), dtype=torch.float, device=x.device)  # (seq_len,)
        rotate = torch.outer(pos, self.thetas)  # (seq_len, dim)

        return x * torch.cos(rotate) + rotate_half(x) * torch.sin(rotate)


if __name__ == "__main__":
    import torch

    x = torch.randn(10, 16)  # (seq_len, dim)
    rope = RotaryPositionalEncoding(dim=16)
    y = rope(x)
    assert y.shape == x.shape, f"y shape {y.shape} must be same as x shape {x.shape}"
