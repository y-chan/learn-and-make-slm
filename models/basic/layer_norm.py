from torch import nn, Tensor

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int | list[int]):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-6

    def forward(self, x: Tensor) -> Tensor:
        x_mean = x.mean(dim=-1, keepdim=True)
        x_var = x.var(dim=-1, keepdim=True)
        x_normalized = (x - x_mean) / torch.sqrt(x_var + self.eps)
        return x_normalized * self.gamma + self.beta


if __name__ == "__main__":
    import torch

    layer_norm = LayerNorm([10])

    x = torch.randn(2, 10)
    assert layer_norm(x).shape == (2, 10)
    print("LayerNorm test passed")
