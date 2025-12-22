from torch import Tensor, nn


class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        # Subtract max for numerical stability
        x_max = x.max(dim=-1, keepdim=True)[0]
        x_exp = (x - x_max).exp()
        return x_exp / x_exp.sum(dim=-1, keepdim=True)
