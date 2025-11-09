from torch import nn, Tensor

class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

if __name__ == "__main__":
    import torch

    softmax = Softmax()

    x = torch.randn(2, 10)
    assert softmax(x).shape == (2, 10)
    print("Softmax test passed")
