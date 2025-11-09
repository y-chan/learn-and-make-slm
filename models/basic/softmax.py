from torch import Tensor, nn


class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        bottom = x.exp().sum(dim=-1, keepdim=True)
        return x.exp() / bottom


if __name__ == "__main__":
    import torch

    softmax = Softmax()

    x = torch.randn(2, 10)
    assert softmax(x).shape == (2, 10)
    print("Softmax test passed")
