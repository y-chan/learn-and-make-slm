from torch import Tensor
import numpy as np
import torch
from utils.pad import pad_1D


def dataset_collate(batch: list[list[int]], torch_convert: bool = True) -> dict[str, np.ndarray] | dict[str, Tensor]:
    lengths = np.array([len(x) for x in batch])
    tokens_ids = pad_1D(batch)

    res = {
        "tokens_ids": tokens_ids,
        "lengths": lengths,
    }

    if torch_convert:
        res = {k: torch.from_numpy(v) for k, v in res.items()}
    return res


def random_end_lengths(lengths: Tensor) -> Tensor:
    end_lengths = (torch.rand_like(lengths, dtype=torch.float) * lengths).long()
    return end_lengths
