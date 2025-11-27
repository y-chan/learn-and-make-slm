import torch
import pytest
from models.transformer.attention import ScaledDotProductAttention, _HAS_XFORMERS


# xformersは通常CUDAでのみ動作し、float16/bfloat16のみをサポート
@pytest.mark.skipif(not _HAS_XFORMERS or not torch.cuda.is_available(), reason="xFormers or CUDA is not available")
def test_scaled_dot_product_attention_without_mask():
    """xformers_forwardとreference_forwardの結果が一致することを確認（マスクなし）"""
    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.float16  # xformersはfloat16/bfloat16のみサポート

    # パラメータ設定
    batch_size = 2
    n_heads = 4
    seq_len = 32  # d_kが32以上必要
    d_k = 64

    # モデル作成
    attention = ScaledDotProductAttention(d_k=d_k).to(device=device, dtype=dtype)

    # 入力データ作成 (B, H, S, D)
    Q = torch.randn(batch_size, n_heads, seq_len, d_k, device=device, dtype=dtype)
    K = torch.randn(batch_size, n_heads, seq_len, d_k, device=device, dtype=dtype)
    V = torch.randn(batch_size, n_heads, seq_len, d_k, device=device, dtype=dtype)

    # xformers_forwardの結果
    with torch.no_grad():
        output_xformers = attention.xformers_forward(Q, K, V, seq_lens=None)

    # reference_forwardの結果
    with torch.no_grad():
        output_reference = attention.reference_forward(Q, K, V, seq_lens=None)

    # 結果の比較（相対誤差と絶対誤差を考慮、float16は精度が低いため緩和）
    assert output_xformers.shape == output_reference.shape, (
        f"Shape mismatch: {output_xformers.shape} vs {output_reference.shape}"
    )

    torch.testing.assert_close(
        output_xformers,
        output_reference,
        rtol=1e-2,  # float16のため緩和
        atol=1e-3,  # float16のため緩和
        msg="xformers_forward and reference_forward outputs do not match (without mask)",
    )


@pytest.mark.skipif(not _HAS_XFORMERS or not torch.cuda.is_available(), reason="xFormers or CUDA is not available")
def test_scaled_dot_product_attention_with_mask():
    """xformers_forwardとreference_forwardの結果が一致することを確認（マスクあり）"""
    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.float16

    # パラメータ設定
    batch_size = 2
    n_heads = 4
    seq_len = 32
    d_k = 64

    # モデル作成
    attention = ScaledDotProductAttention(d_k=d_k).to(device=device, dtype=dtype)

    # 入力データ作成 (B, H, S, D)
    Q = torch.randn(batch_size, n_heads, seq_len, d_k, device=device, dtype=dtype)
    K = torch.randn(batch_size, n_heads, seq_len, d_k, device=device, dtype=dtype)
    V = torch.randn(batch_size, n_heads, seq_len, d_k, device=device, dtype=dtype)

    # seq_lens（各バッチのシーケンス長）
    seq_lens = torch.tensor([24, 30], device=device)  # 最初のバッチは24トークン、2番目は30トークン

    # xformers_forwardの結果
    with torch.no_grad():
        output_xformers = attention.xformers_forward(Q, K, V, seq_lens=seq_lens)

    # reference_forwardの結果
    with torch.no_grad():
        output_reference = attention.reference_forward(Q, K, V, seq_lens=seq_lens)

    # 結果の比較（相対誤差と絶対誤差を考慮、float16は精度が低いため緩和）
    assert output_xformers.shape == output_reference.shape, (
        f"Shape mismatch: {output_xformers.shape} vs {output_reference.shape}"
    )

    # パディング部分を除いた有効部分のみを比較
    for batch_idx, length in enumerate(seq_lens):
        torch.testing.assert_close(
            output_xformers[batch_idx, :, :length, :],
            output_reference[batch_idx, :, :length, :],
            rtol=1e-2,  # float16のため緩和
            atol=1e-3,  # float16のため緩和
            msg=f"xformers_forward and reference_forward outputs do not match for batch {batch_idx} (with mask)",
        )


@pytest.mark.skipif(not _HAS_XFORMERS or not torch.cuda.is_available(), reason="xFormers or CUDA is not available")
def test_scaled_dot_product_attention_different_sizes():
    """様々なサイズで結果が一致することを確認"""
    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.float16

    test_cases = [
        (1, 2, 32, 64),  # 小さいケース
        (2, 4, 64, 64),  # 中程度
        (4, 8, 128, 128),  # 大きめ
    ]

    for batch_size, n_heads, seq_len, d_k in test_cases:
        attention = ScaledDotProductAttention(d_k=d_k).to(device=device, dtype=dtype)

        Q = torch.randn(batch_size, n_heads, seq_len, d_k, device=device, dtype=dtype)
        K = torch.randn(batch_size, n_heads, seq_len, d_k, device=device, dtype=dtype)
        V = torch.randn(batch_size, n_heads, seq_len, d_k, device=device, dtype=dtype)

        with torch.no_grad():
            output_xformers = attention.xformers_forward(Q, K, V, seq_lens=None)
            output_reference = attention.reference_forward(Q, K, V, seq_lens=None)

        torch.testing.assert_close(
            output_xformers,
            output_reference,
            rtol=1e-2,  # float16のため緩和
            atol=1e-3,  # float16のため緩和
            msg=f"Outputs do not match for size (B={batch_size}, H={n_heads}, S={seq_len}, D={d_k})",
        )
