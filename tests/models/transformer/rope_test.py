import torch
import math
import pytest

from models.transformer.rope import RotaryPositionalEmbedding, rotate_half


def test_rope_shape():
    """形状が保存されることを確認"""
    x = torch.randn(10, 16)  # (seq_len, dim)
    rope = RotaryPositionalEmbedding(dim=16)
    y = rope(x)

    assert y.shape == x.shape


# 以下AIが作成したテスト
def test_rope_manual_calculation():
    """小さな例で手計算した結果と比較"""
    dim = 4
    seq_len = 2
    base = 10_000

    # 簡単な入力を用意
    x = torch.ones(seq_len, dim)

    # 手動で期待値を計算
    # theta_i = 1 / (base^(2i/dim)) for i in [0, 1]
    # theta_0 = 1.0 / (base ** (0.0 / dim))  # = 1.0
    theta_1 = 1.0 / (base ** (2.0 / dim))  # = 1 / base^0.5

    # 各位置での回転角度 = pos * theta
    # pos=0: angles = [0, 0, 0, 0]
    # pos=1: angles = [theta_0, theta_0, theta_1, theta_1] = [1.0, 1.0, theta_1, theta_1]

    expected = torch.zeros(seq_len, dim)

    # pos=0の場合: cos(0)=1, sin(0)=0
    # x * cos + rotate_half(x) * sin = [1,1,1,1] * 1 + [-1,1,-1,1] * 0 = [1,1,1,1]
    expected[0] = torch.tensor([1.0, 1.0, 1.0, 1.0])

    # pos=1の場合:
    # x = [1, 1, 1, 1]
    # rotate_half(x) = [-1, 1, -1, 1]  # (x2の符号反転, x1) のペア
    # cos = [cos(1.0), cos(1.0), cos(theta_1), cos(theta_1)]
    # sin = [sin(1.0), sin(1.0), sin(theta_1), sin(theta_1)]
    # result = x * cos + rotate_half(x) * sin
    #        = [1, 1, 1, 1] * cos + [-1, 1, -1, 1] * sin
    #        = [cos - sin, cos + sin, cos - sin, cos + sin]
    cos_1 = math.cos(1.0)
    sin_1 = math.sin(1.0)
    cos_theta_1 = math.cos(theta_1)
    sin_theta_1 = math.sin(theta_1)
    expected[1] = torch.tensor([cos_1 - sin_1, cos_1 + sin_1, cos_theta_1 - sin_theta_1, cos_theta_1 + sin_theta_1])

    # 実装をテスト
    rope = RotaryPositionalEmbedding(dim=dim)
    result = rope(x)

    assert torch.allclose(result, expected, atol=1e-6), f"Expected:\n{expected}\nGot:\n{result}"


def test_rope_norm_preservation():
    """回転操作なのでノルムがほぼ保存されることを確認"""
    x = torch.randn(10, 16)  # (seq_len, dim)
    rope = RotaryPositionalEmbedding(dim=16)
    y = rope(x)

    # RoPEは回転操作なので、各ベクトルのノルムは保存される
    x_norms = torch.norm(x, dim=-1)
    y_norms = torch.norm(y, dim=-1)

    assert torch.allclose(x_norms, y_norms, atol=1e-5)


def test_rope_relative_position():
    """相対位置が同じなら内積も同じになることを確認"""
    dim = 16
    rope = RotaryPositionalEmbedding(dim=dim)

    # 同じベクトルペアなら相対位置が同じなら内積も同じになるはず
    q_same = torch.ones(4, dim)
    k_same = torch.ones(4, dim) * 2

    q_same_rope = rope(q_same)
    k_same_rope = rope(k_same)

    # 相対位置が同じペアの内積を比較
    # (pos=0, pos=1) vs (pos=2, pos=3) -> どちらも相対位置 = 1
    dot_same_01 = torch.dot(q_same_rope[0], k_same_rope[1])
    dot_same_23 = torch.dot(q_same_rope[2], k_same_rope[3])

    # 相対位置が同じなら内積も同じになる
    assert torch.allclose(dot_same_01, dot_same_23, atol=1e-5)


def test_rotate_half():
    """rotate_half関数の動作を確認"""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # (1, 4)
    # rotate_half: [x1, x2, x3, x4] -> [-x2, x1, -x4, x3]
    expected = torch.tensor([[-2.0, 1.0, -4.0, 3.0]])

    result = rotate_half(x)
    assert torch.allclose(result, expected)


def test_yarn_shape():
    """YaRNでも形状が保存されることを確認"""
    x = torch.randn(10, 64)  # (seq_len, dim)
    yarn = RotaryPositionalEmbedding(dim=64, max_seq_len=2048, scale_factor=4.0)
    y = yarn(x)

    assert y.shape == x.shape


def test_yarn_attention_scale():
    """attention_scaleが正しく計算されることを確認"""
    # sqrt(1/t) = 0.1 * ln(s) + 1
    scale_factor = 4.0
    expected_attention_scale = 0.1 * math.log(scale_factor) + 1.0

    yarn = RotaryPositionalEmbedding(dim=64, max_seq_len=2048, scale_factor=scale_factor)

    assert abs(yarn.attention_scale - expected_attention_scale) < 1e-6


def test_yarn_no_attention_scale_when_no_extension():
    """scale_factor=1.0のときはattention_scale=1.0"""
    rope = RotaryPositionalEmbedding(dim=64, max_seq_len=2048, scale_factor=1.0)

    assert rope.attention_scale == 1.0


def test_yarn_ramp_function_boundaries():
    """ランプ関数が境界で正しい値を返すことを確認"""
    dim = 64
    yarn = RotaryPositionalEmbedding(dim=dim, max_seq_len=2048, scale_factor=4.0)

    # ランプ関数をテスト
    r = torch.arange(0, dim // 2, dtype=torch.float32)
    gamma = yarn._compute_ramp_function(r)

    # gammaは0から1の範囲内
    assert torch.all(gamma >= 0.0)
    assert torch.all(gamma <= 1.0)

    # 最初の方（高周波）はgamma ≈ 0
    assert gamma[0] < 0.5

    # 最後の方（低周波）はgamma ≈ 1
    assert gamma[-1] > 0.5


def test_yarn_interpolation_high_freq_preserved():
    """高周波次元（小さいインデックス）で元のthetaが保持されることを確認"""
    dim = 64
    scale_factor = 4.0

    yarn = RotaryPositionalEmbedding(dim=dim, max_seq_len=2048, scale_factor=scale_factor)
    rope = RotaryPositionalEmbedding(dim=dim, max_seq_len=2048, scale_factor=1.0)

    # 高周波次元（インデックス0, 1）のthetaを比較
    # YaRNでも高周波は補間されないので、RoPEと同じはず（attention_scaleの違いを除く）
    # cos/sinのインデックス0, 1は最初のtheta（高周波）に対応

    # attention_scaleで正規化して比較
    yarn_cos_normalized = yarn.cos / yarn.attention_scale
    rope_cos_normalized = rope.cos / rope.attention_scale

    # 高周波次元（インデックス0, 1）は同じはず
    assert torch.allclose(yarn_cos_normalized[:, 0], rope_cos_normalized[:, 0], atol=1e-5)
    assert torch.allclose(yarn_cos_normalized[:, 1], rope_cos_normalized[:, 1], atol=1e-5)


def test_yarn_interpolation_low_freq_scaled():
    """低周波次元（大きいインデックス）でthetaがスケールされることを確認"""
    dim = 64
    scale_factor = 4.0

    yarn = RotaryPositionalEmbedding(dim=dim, max_seq_len=2048, scale_factor=scale_factor)
    rope = RotaryPositionalEmbedding(dim=dim, max_seq_len=2048, scale_factor=1.0)

    # 低周波次元（最後のインデックス）のthetaを比較
    # YaRNでは低周波はθ/sにスケールされるので、cos/sinの周期が長くなる
    # つまり、同じ位置でのcos値が1に近くなる（回転が遅くなる）

    # attention_scaleで正規化
    yarn_cos_normalized = yarn.cos / yarn.attention_scale
    rope_cos_normalized = rope.cos / rope.attention_scale

    # 低周波次元（最後の2つ）は異なるはず
    last_dim = dim - 2
    assert not torch.allclose(yarn_cos_normalized[:, last_dim], rope_cos_normalized[:, last_dim], atol=1e-3)


def test_yarn_differs_from_rope():
    """YaRNとRoPEで異なる結果が出ることを確認"""
    x = torch.randn(10, 64)

    yarn = RotaryPositionalEmbedding(dim=64, max_seq_len=2048, scale_factor=4.0)
    rope = RotaryPositionalEmbedding(dim=64, max_seq_len=2048, scale_factor=1.0)

    y_yarn = yarn(x)
    y_rope = rope(x)

    # YaRNとRoPEは異なる結果を返すはず
    assert not torch.allclose(y_yarn, y_rope, atol=1e-3)


def test_yarn_relative_position():
    """YaRNでも相対位置が同じなら内積も同じになることを確認"""
    dim = 64
    yarn = RotaryPositionalEmbedding(dim=dim, max_seq_len=2048, scale_factor=4.0)

    q = torch.ones(4, dim)
    k = torch.ones(4, dim) * 2

    q_yarn = yarn(q)
    k_yarn = yarn(k)

    # 相対位置が同じペアの内積を比較
    dot_01 = torch.dot(q_yarn[0], k_yarn[1])
    dot_23 = torch.dot(q_yarn[2], k_yarn[3])

    assert torch.allclose(dot_01, dot_23, atol=1e-5)


def test_yarn_norm_scaling():
    """YaRNではノルムがattention_scaleでスケールされることを確認"""
    x = torch.randn(10, 64)
    yarn = RotaryPositionalEmbedding(dim=64, max_seq_len=2048, scale_factor=4.0)
    y = yarn(x)

    x_norms = torch.norm(x, dim=-1)
    y_norms = torch.norm(y, dim=-1)

    # YaRNではノルムがattention_scaleでスケールされる
    expected_norms = x_norms * yarn.attention_scale

    assert torch.allclose(y_norms, expected_norms, atol=1e-4)


def test_yarn_dynamic_scaling():
    """動的スケーリングが正しく動作することを確認"""
    dim = 64
    original_max_seq_len = 2048

    yarn = RotaryPositionalEmbedding(
        dim=dim,
        max_seq_len=original_max_seq_len,
        scale_factor=4.0,
        enable_dynamic_scaling=True,
    )

    # 元のmax_seq_lenより長いシーケンス
    long_seq_len = 4096
    x = torch.randn(long_seq_len, dim)

    # エラーなく動作すること
    y = yarn(x)
    assert y.shape == x.shape

    # dynamic_scaleが更新されていること
    expected_dynamic_scale = long_seq_len / original_max_seq_len
    expected_attention_scale = 0.1 * math.log(expected_dynamic_scale) + 1.0
    assert abs(yarn.attention_scale - expected_attention_scale) < 1e-6


def test_yarn_cache_update():
    """キャッシュが正しく更新されることを確認"""
    dim = 64
    yarn = RotaryPositionalEmbedding(dim=dim, max_seq_len=1024, scale_factor=4.0)

    # 初期キャッシュサイズ
    assert yarn.cos.shape[0] == 1024

    # より長いシーケンスを処理
    x = torch.randn(2048, dim)
    _ = yarn(x)

    # キャッシュが拡張されていること
    assert yarn.cos.shape[0] >= 2048


@pytest.mark.parametrize("scale_factor", [2.0, 4.0, 8.0, 16.0])
def test_yarn_various_scale_factors(scale_factor):
    """様々なscale_factorで正しく動作することを確認"""
    dim = 64
    yarn = RotaryPositionalEmbedding(dim=dim, max_seq_len=2048, scale_factor=scale_factor)

    x = torch.randn(10, dim)
    y = yarn(x)

    # 形状が保存される
    assert y.shape == x.shape

    # attention_scaleが正しい
    expected_scale = 0.1 * math.log(scale_factor) + 1.0
    assert abs(yarn.attention_scale - expected_scale) < 1e-6
