from typing import TypedDict
import datasets
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np
import torch
from utils.pad import pad_1D


class SimpleStoriesBatchNumpy(TypedDict):
    tokens_ids: np.ndarray
    lengths: np.ndarray


class SimpleStoriesBatchTorch(TypedDict):
    tokens_ids: Tensor
    lengths: Tensor


SimpleStoriesBatch = SimpleStoriesBatchNumpy | SimpleStoriesBatchTorch


class SimpleStoriesBothDataset(Dataset):
    def __init__(self, subset: str = "train"):
        self.dataset_ja = datasets.load_dataset("SimpleStories/SimpleStories-JA", split=subset)
        self.dataset_en = datasets.load_dataset("SimpleStories/SimpleStories", split=subset)
        self.dataset_ja_len = len(self.dataset_ja)
        self.dataset_en_len = len(self.dataset_en)

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


def dataset_collate(batch, torch_convert: bool = True, max_length: int = 512) -> SimpleStoriesBatch:
    # max_lengthを超えるサンプルについては、ランダムな開始地点からカットする
    stories = []
    for x in batch:
        story = np.array(x["story"], dtype=np.int64)
        if len(story) > max_length:
            # ランダムな開始地点を決定（0からlen(story) - max_lengthの間）
            start_idx = int(torch.randint(0, len(story) - max_length + 1, (1,)).item())
            story = story[start_idx : start_idx + max_length]
        stories.append(story)

    lengths = np.array([len(story) for story in stories])
    tokens_ids = pad_1D(stories)

    res = {
        "tokens_ids": tokens_ids,
        "lengths": lengths,
    }

    if torch_convert:
        res = {k: torch.from_numpy(v) for k, v in res.items()}
    return res
