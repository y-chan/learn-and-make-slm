import torch
from models.transformer.decoder_layer import DecoderLayer


def test_decoder_layer_shape():
    """DecoderLayerの出力計上を確認"""
    batch_size = 2
    seq_len = 32
    d_model = 64
    n_heads = 4
    n_groups = 2

    x = torch.rand(batch_size, seq_len, d_model)

    decoder_layer = DecoderLayer(d_model=d_model, n_heads=n_heads, n_groups=n_groups)
    output = decoder_layer(x, seq_lens=torch.tensor([seq_len]))

    assert output.shape == (batch_size, seq_len, d_model)
