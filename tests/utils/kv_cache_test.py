import pytest
import torch
from utils.kv_cache import CacheEntry, KVCache


def test_kv_cache_append_and_length():
    kv_cache = KVCache()

    key1 = torch.randn(1, 4, 8, 4)  # (batch_size, n_heads, seq_len, d_k)
    value1 = torch.randn(1, 4, 8, 4)  # (batch_size, n_heads, seq_len, d_k)
    idx1 = kv_cache.append(CacheEntry, key1, value1)
    assert idx1 == 0
    assert len(kv_cache) == 1

    key2 = torch.randn(1, 2, 8, 4)
    value2 = torch.randn(1, 2, 8, 4)
    idx2 = kv_cache.append(CacheEntry, key2, value2)
    assert idx2 == 1
    assert len(kv_cache) == 2


def test_kv_cache_update_reset():
    kv_cache = KVCache()

    seq_len_1 = 8
    seq_len_2 = 4

    key = torch.randn(1, 4, seq_len_1, 4)  # (batch_size, n_heads, seq_len, d_k)
    value = torch.randn(1, 4, seq_len_1, 4)
    idx = kv_cache.append(CacheEntry, key, value)

    new_key = torch.randn(1, 4, seq_len_2, 4)
    new_value = torch.randn(1, 4, seq_len_2, 4)
    updated_key, updated_value = kv_cache.update(idx, new_key, new_value)

    assert updated_key.shape == (1, 4, seq_len_1 + seq_len_2, 4)
    assert updated_value.shape == (1, 4, seq_len_1 + seq_len_2, 4)

    kv_cache.reset(idx)
    assert torch.all(kv_cache.cache[idx].key == 0)
    assert torch.all(kv_cache.cache[idx].value == 0)


def test_kv_cache_offload_prefetch_cpu_roundtrip():
    torch.manual_seed(0)
    kv_cache = KVCache()

    key = torch.randn(1, 2, 3, 4)
    value = torch.randn(1, 2, 3, 4)
    original_key = key.clone()
    original_value = value.clone()

    idx = kv_cache.append(CacheEntry, key, value)

    kv_cache.offload(idx)
    assert kv_cache.cache[idx].key.device.type == "cpu"
    assert kv_cache.cache[idx].value.device.type == "cpu"

    kv_cache.prefetch(idx)
    assert kv_cache.cache[idx].key.device.type == original_key.device.type
    assert kv_cache.cache[idx].value.device.type == original_value.device.type

    torch.testing.assert_close(kv_cache.cache[idx].key, original_key)
    torch.testing.assert_close(kv_cache.cache[idx].value, original_value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_kv_cache_offload_prefetch_cuda_roundtrip():
    torch.manual_seed(0)
    kv_cache = KVCache()

    device = torch.device("cuda")
    key = torch.randn(1, 2, 3, 4, device=device)
    value = torch.randn(1, 2, 3, 4, device=device)
    original_key = key.clone()
    original_value = value.clone()

    idx = kv_cache.append(CacheEntry, key, value)

    kv_cache.offload(idx)
    assert kv_cache.cache[idx].key.device.type == "cpu"
    assert kv_cache.cache[idx].value.device.type == "cpu"

    kv_cache.prefetch(idx)
    assert kv_cache.cache[idx].key.device == device
    assert kv_cache.cache[idx].value.device == device

    torch.testing.assert_close(kv_cache.cache[idx].key, original_key)
    torch.testing.assert_close(kv_cache.cache[idx].value, original_value)
