from torch import nn, Tensor
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        i = torch.arange(self.d_model, dtype=torch.float).unsqueeze(0)
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / self.d_model)
        pe = pos * angle_rates
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        pe = pe.unsqueeze(0)
        return x + pe
