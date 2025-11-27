from beartype import beartype
from jaxtyping import jaxtyped, Float
from torch import Tensor, nn
import torch


@jaxtyped(typechecker=beartype)
def rotate_half(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    x = x.reshape(*x.shape[:-1], -1, 2)  # [..., dim/2, 2]
    x1, x2 = x[..., 0], x[..., 1]
    rotated = torch.stack((-x2, x1), dim=-1)  # [..., dim/2, 2]
    return rotated.reshape(*x.shape[:-2], -1)  # [..., dim]


class RotaryPositionalEncoding(nn.Module):
    thetas: Tensor  # (dim,)

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        _thetas = self._setup_thetas(dim)
        self.register_buffer("thetas", _thetas)

    @jaxtyped(typechecker=beartype)
    def _setup_thetas(self, dim: int) -> Float[Tensor, "{dim}"]:  # noqa: F821
        base = 10000
        vec = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        return torch.repeat_interleave(vec, repeats=2)

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "seq {self.dim}"]) -> Float[Tensor, "seq {self.dim}"]:  # noqa: F821
        start = 1
        end = start + x.size(-2)
        pos = torch.arange(start, end, dtype=torch.float, device=x.device)  # (seq,)
        rotate = torch.outer(pos, self.thetas)  # (seq, dim)

        return x * torch.cos(rotate) + rotate_half(x) * torch.sin(rotate)


if __name__ == "__main__":
    import torch

    x = torch.randn(10, 16)  # (seq_len, dim)
    rope = RotaryPositionalEncoding(dim=16)
    y = rope(x)
    assert y.shape == x.shape, f"y shape {y.shape} must be same as x shape {x.shape}"
