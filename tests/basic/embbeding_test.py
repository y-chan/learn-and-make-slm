import torch

from models.basic.embedding import Embedding


def test_embedding_gathers_rows():
    embedding = Embedding(50, 20)
    x = torch.randint(0, 50, (2, 10))

    y = embedding(x)

    assert y.shape == (2, 10, 20)


def test_embedding_output():
    embedding = Embedding(50, 20)
    x = torch.randint(0, 50, (2, 10))

    y = embedding(x)

    y_ref = torch.nn.functional.embedding(x, embedding.w)
    torch.testing.assert_close(y, y_ref)
