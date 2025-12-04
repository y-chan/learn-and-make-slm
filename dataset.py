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
    # 本来であれば、Transformerの学習においては上三角行列を生成し、
    # それをマスクとして用いることで短い系列も学習要素として含めるが、
    # 今回はFlash Attentionのために特殊なマスクを使用するため、
    # lengthsを、50%の確率でランダムな長さに変更する形を取る。
    with torch.no_grad():
        end_lengths = (torch.rand_like(lengths, dtype=torch.float) * lengths).long()
        mask = torch.rand_like(lengths, dtype=torch.float) < 0.5
        end_lengths = end_lengths * mask + lengths * ~mask
    return end_lengths
