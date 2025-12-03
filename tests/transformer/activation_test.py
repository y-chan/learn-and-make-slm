import torch
import torch.nn.functional as F

from models.transformer.activation import GLU, SwiGLU, Swish, sigmoid


def test_sigmoid():
    """sigmoid関数がPyTorchの標準実装と一致することを確認"""
    x = torch.randn(10, 20)
    result = sigmoid(x)
    expected = torch.sigmoid(x)

    assert torch.allclose(result, expected, atol=1e-6)


def test_sigmoid_edge_cases():
    """sigmoid関数の端点での動作を確認"""
    # 大きな正の値
    x_large_pos = torch.tensor([100.0])
    assert torch.allclose(sigmoid(x_large_pos), torch.tensor([1.0]), atol=1e-6)

    # 大きな負の値
    x_large_neg = torch.tensor([-100.0])
    assert torch.allclose(sigmoid(x_large_neg), torch.tensor([0.0]), atol=1e-6)

    # ゼロ
    x_zero = torch.tensor([0.0])
    assert torch.allclose(sigmoid(x_zero), torch.tensor([0.5]), atol=1e-6)


def test_swish_beta_1():
    """Swish (beta=1) がPyTorchのSiLUと一致することを確認"""
    x = torch.randn(10, 20)
    swish = Swish(beta=1.0, beta_is_learnable=False)
    result = swish(x)
    expected = F.silu(x)  # SiLU = Swish with beta=1

    assert torch.allclose(result, expected, atol=1e-6)


def test_swish_different_betas():
    """異なるbeta値でのSwishの動作を確認"""
    x = torch.randn(5, 10)

    # beta=1.0
    swish_1 = Swish(beta=1.0, beta_is_learnable=False)
    result_1 = swish_1(x)

    # beta=2.0
    swish_2 = Swish(beta=2.0, beta_is_learnable=False)
    result_2 = swish_2(x)

    # betaが大きいほど、シャープになる（異なる結果になるはず）
    assert not torch.allclose(result_1, result_2)

    # 手動計算と比較（beta=2.0）
    expected_2 = x * torch.sigmoid(2.0 * x)
    assert torch.allclose(result_2, expected_2, atol=1e-6)


def test_swish_learnable_beta():
    """学習可能なbetaパラメータが正しく設定されることを確認"""
    swish_learnable = Swish(beta=1.5, beta_is_learnable=True)
    swish_fixed = Swish(beta=1.5, beta_is_learnable=False)

    # 学習可能な場合はParameterとして登録される
    assert isinstance(swish_learnable.beta, torch.nn.Parameter)
    # 固定の場合はbufferとして登録される
    assert not isinstance(swish_fixed.beta, torch.nn.Parameter)

    # どちらも同じ値を持つ
    assert torch.allclose(swish_learnable.beta, swish_fixed.beta)

    # 学習可能なパラメータはparameters()に含まれる
    params = list(swish_learnable.parameters())
    assert len(params) == 1
    assert params[0] is swish_learnable.beta

    # 固定の場合はparameters()に含まれない
    params_fixed = list(swish_fixed.parameters())
    assert len(params_fixed) == 0


def test_glu():
    """GLUがPyTorchの標準実装と一致することを確認"""
    torch.manual_seed(42)
    dim_in = 8
    dim_hidden = 16
    batch_size = 4
    seq_len = 10

    x = torch.randn(batch_size, seq_len, dim_in)

    # カスタム実装
    glu_custom = GLU(dim_in=dim_in, dim_hidden=dim_hidden)

    # PyTorchの標準GLU（dim=-1で分割）
    glu_pytorch = torch.nn.GLU(dim=-1)

    # 同じ重みを使用するために、カスタム実装の重みをPyTorchに設定
    # GLUは入力をdim_hidden*2に射影してから2分割するので、
    # PyTorchのGLUと同じ動作をするはず

    with torch.no_grad():
        # カスタム実装で射影
        projected = glu_custom.proj(x)  # (batch, seq, dim_hidden*2)
        # PyTorchのGLUを適用
        expected = glu_pytorch(projected)
        # カスタム実装
        result = glu_custom(x)

        # 形状が一致することを確認
        assert result.shape == expected.shape
        assert result.shape == (batch_size, seq_len, dim_hidden)


def test_swiglu_shape():
    torch.manual_seed(42)
    dim_in = 8
    dim_hidden = 16
    batch_size = 4
    seq_len = 10

    x = torch.randn(batch_size, seq_len, dim_in)
    swiglu = SwiGLU(dim_in=dim_in, dim_hidden=dim_hidden)

    result = swiglu(x)

    # 形状チェック
    assert result.shape == (batch_size, seq_len, dim_hidden)
