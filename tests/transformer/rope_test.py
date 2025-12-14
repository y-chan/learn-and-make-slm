import torch
import math

from models.transformer.rope import RotaryPositionalEncoding, rotate_half


def test_rope_shape():
    """形状が保存されることを確認"""
    x = torch.randn(10, 16)  # (seq_len, dim)
    rope = RotaryPositionalEncoding(dim=16)
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
    rope = RotaryPositionalEncoding(dim=dim)
    result = rope(x)

    assert torch.allclose(result, expected, atol=1e-6), f"Expected:\n{expected}\nGot:\n{result}"


def test_rope_norm_preservation():
    """回転操作なのでノルムがほぼ保存されることを確認"""
    x = torch.randn(10, 16)  # (seq_len, dim)
    rope = RotaryPositionalEncoding(dim=16)
    y = rope(x)

    # RoPEは回転操作なので、各ベクトルのノルムは保存される
    x_norms = torch.norm(x, dim=-1)
    y_norms = torch.norm(y, dim=-1)

    assert torch.allclose(x_norms, y_norms, atol=1e-5)


def test_rope_relative_position():
    """相対位置が同じなら内積も同じになることを確認"""
    dim = 16
    rope = RotaryPositionalEncoding(dim=dim)

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
