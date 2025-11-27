from torch import nn, Tensor
import torch


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.one_hot_vec = torch.eye(num_embeddings)
        self.w = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, x: Tensor) -> Tensor:
        one_hot = self.one_hot_vec[x]
        return one_hot @ self.w
