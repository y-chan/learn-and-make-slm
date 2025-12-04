from typing import TypedDict

import datasets
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np
import torch
from utils.pad import pad_1D


class SimpleStoriesBothDataset(Dataset):
    def __init__(self, subset: str = "train"):
        self.dataset_ja = datasets.load_dataset("SimpleStories/SimpleStories-JA", split=subset)
        self.dataset_en = datasets.load_dataset("SimpleStories/SimpleStories", split=subset)
        self.dataset_ja_len = len(self.dataset_ja)
        self.dataset_en_len = len(self.dataset_en)
        self.dataset_en_len = 0

    def __len__(self):
        return self.dataset_ja_len + self.dataset_en_len

    def __getitem__(self, idx):
        if idx < self.dataset_ja_len:
            return self.dataset_ja[idx]
        else:
            return self.dataset_en[idx - self.dataset_ja_len]

    def map(self, func, batched: bool = False, **kwargs):
        self.dataset_ja = self.dataset_ja.map(func, batched=batched, **kwargs)
        self.dataset_en = self.dataset_en.map(func, batched=batched, **kwargs)
        return self


def dataset_collate(batch, torch_convert: bool = True) -> dict[str, np.ndarray] | dict[str, Tensor]:
    lengths = np.array([len(x["story"]) for x in batch])
    tokens_ids = pad_1D([np.array(x["story"], dtype=np.int64) for x in batch])

    res = {
        "tokens_ids": tokens_ids,
        "lengths": lengths,
    }

    if torch_convert:
        res = {k: torch.from_numpy(v) for k, v in res.items()}
    return res


def random_end_lengths(lengths: Tensor) -> Tensor:
    # 本来であれば、Transformerの学習においては上三角行列を生成し、
    # それをマスクとして用いることで短い系列も学習要素として含めるが、
    # 今回はFlash Attentionのために特殊なマスクを使用するため、
    # lengthsを、50%の確率でランダムな長さに変更する形を取る。
    with torch.no_grad():
        end_lengths = (torch.rand_like(lengths, dtype=torch.float) * lengths).long()
        mask = torch.rand_like(lengths, dtype=torch.float) < 0.5
        end_lengths = end_lengths * mask + lengths * ~mask
    return end_lengths
