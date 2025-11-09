from torch import nn, Tensor
import torch

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.w = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, x: Tensor) -> Tensor:
        one_hot = self.one_hot(x)
        return one_hot @ self.w
    
    def one_hot(self, x: Tensor) -> Tensor:
        num_classes = self.num_embeddings
        one_hot = torch.eye(num_classes)[x]
        return one_hot.float()


if __name__ == "__main__":
    import torch

    embedding = Embedding(50, 20)

    x = torch.randint(0, 50, (2, 10))
    assert embedding(x).shape == (2, 10, 20)
    print("Embedding test passed")
