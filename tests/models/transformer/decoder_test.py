import torch

from models.transformer.decoder import Decoder


def test_decoder_shape():
    """Decoderの順伝播の出力形状を確認"""
    batch_size = 2
    seq_len = 32
    n_layers = 2
    d_model = 64
    n_heads = 4
    n_groups = 2
    n_vocab = 20
    end_token_id = 19

    x = torch.randint(0, n_vocab, (batch_size, seq_len))

    decoder = Decoder(
        n_vocab=n_vocab, n_layers=n_layers, d_model=d_model, n_heads=n_heads, n_groups=n_groups, end_token_id=end_token_id
    )
    output = decoder(x, seq_lens=torch.tensor([seq_len, seq_len]))

    assert output.shape == (batch_size, seq_len, n_vocab)
