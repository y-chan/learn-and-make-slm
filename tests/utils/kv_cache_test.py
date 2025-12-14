import torch
from utils.kv_cache import KVCache


def test_kv_cache_append_and_length():
    kv_cache = KVCache()

    key1 = torch.randn(1, 4, 8, 4)  # (batch_size, n_heads, seq_len, d_k)
    value1 = torch.randn(1, 4, 8, 4)  # (batch_size, n_heads, seq_len, d_k)
    idx1 = kv_cache.append(key1, value1)
    assert idx1 == 0
    assert len(kv_cache) == 1

    key2 = torch.randn(1, 2, 8, 4)
    value2 = torch.randn(1, 2, 8, 4)
    idx2 = kv_cache.append(key2, value2)
    assert idx2 == 1
    assert len(kv_cache) == 2


def test_kv_cache_update_reset():
    kv_cache = KVCache()

    seq_len_1 = 8
    seq_len_2 = 4

    key = torch.randn(1, 4, seq_len_1, 4)  # (batch_size, n_heads, seq_len, d_k)
    value = torch.randn(1, 4, seq_len_1, 4)
    idx = kv_cache.append(key, value)

    new_key = torch.randn(1, 4, seq_len_2, 4)
    new_value = torch.randn(1, 4, seq_len_2, 4)
    updated_key, updated_value = kv_cache.update(idx, new_key, new_value)

    assert updated_key.shape == (1, 4, seq_len_1 + seq_len_2, 4)
    assert updated_value.shape == (1, 4, seq_len_1 + seq_len_2, 4)

    kv_cache.reset(idx)
    assert torch.all(kv_cache.cache[idx].key == 0)
    assert torch.all(kv_cache.cache[idx].value == 0)
