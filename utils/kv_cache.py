# This code is based on Hugging Face Transformers' cache utilities.
# Original source:
# https://github.com/huggingface/transformers/blob/40dc11cd3eb4126652aa41ef8272525affd4a636/src/transformers/cache_utils.py
#
# Copyright 2018- The Hugging Face team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod

import torch
from jaxtyping import Float
from torch import Tensor


class Cache(ABC):
    def __init__(self) -> None:
        self.key: torch.Tensor
        self.value: torch.Tensor
        self.dtype: torch.dtype
        self.device: torch.device

    def offload(self) -> None:
        """
        Move the key and value tensors to CPU memory.
        """
        self.key = self.key.to("cpu", non_blocking=True)
        self.value = self.value.to("cpu", non_blocking=True)

    def prefetch(self) -> None:
        """
        Prefetch the key and value tensors to their original device.
        """
        self.key = self.key.to(self.device, non_blocking=True)
        self.value = self.value.to(self.device, non_blocking=True)

    def reset(self) -> None:
        self.key.zero_()
        self.value.zero_()

    @abstractmethod
    def update(
        self, new_key: Float[Tensor, "B H S_new D"], new_value: Float[Tensor, "B H S_new D"]
    ) -> tuple[Float[Tensor, "B H S_total D"], Float[Tensor, "B H S_total D"]]:
        """
        Update the cache with new key and value tensors.

        Returns
        -------
        tuple[Float[Tensor, "B H S_total D"], Float[Tensor, "B H S_total D"]]
            The updated key and value tensors.
        """
        ...


class CacheEntry(Cache):
    def __init__(self, key: Float[Tensor, "B H S D"], value: Float[Tensor, "B H S D"]) -> None:
        self.key = key
        self.value = value
        self.dtype = key.dtype
        self.device = key.device

    def update(
        self, new_key: Float[Tensor, "B H S_new D"], new_value: Float[Tensor, "B H S_new D"]
    ) -> tuple[Float[Tensor, "B H S_total D"], Float[Tensor, "B H S_total D"]]:
        self.key = torch.cat([self.key, new_key], dim=-2)
        self.value = torch.cat([self.value, new_value], dim=-2)
        return self.key, self.value


class KVCache:
    def __init__(self) -> None:
        self.cache: list[Cache] = []

    def __len__(self) -> int:
        return len(self.cache)

    def offload(self) -> None:
        for entry in self.cache:
            entry.offload()

    def prefetch(self) -> None:
        for entry in self.cache:
            entry.prefetch()

    def reset(self) -> None:
        for entry in self.cache:
            entry.reset()

    def append(self, key: Float[Tensor, "B H S D"], value: Float[Tensor, "B H S D"]) -> int:
        """
        Append a KV pair and return its index.

        Parameters
        ----------
        key : Float[Tensor, "B H S D"]
        value : Float[Tensor, "B H S D"]

        Returns
        -------
        int
            The index of the appended KV pair.
        """

        self.cache.append(CacheEntry(key, value))
        return len(self.cache) - 1

    def update(
        self, index: int, new_key: Float[Tensor, "B H S_new D"], new_value: Float[Tensor, "B H S_new D"]
    ) -> tuple[Float[Tensor, "B H S_total D"], Float[Tensor, "B H S_total D"]]:
        """
        Concatenate new key and value tensors to the existing ones at the specified index.

        Parameters
        ----------
        index : int
            The index of the KV pair to update.
        new_key : Float[Tensor, "B H S_new D"]
        new_value : Float[Tensor, "B H S_new D"]

        Returns
        -------
        tuple[Float[Tensor, "B H S_total D"], Float[Tensor, "B H S_total D"]]
            The updated key and value tensors.
        """
        # TODO: index existence check
        return self.cache[index].update(new_key, new_value)
