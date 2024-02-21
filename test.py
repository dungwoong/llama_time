import torch
from nodules import attention, rmsnorm, model, wordpiece


def test_attention_shape():
    x = torch.randn((64, 5, 32))  # B x S x H
    attn = attention.RoPESelfAttention(32, 10)
    y = attn(x)
    assert tuple(y.shape) == (64, 5, 32)


def test_rms_norm_shape():
    norm = rmsnorm.RMSNorm(32)
    x = torch.randn((64, 5, 32))
    y = norm(x)
    assert tuple(y.shape) == (64, 5, 32)


def test_simple_model():
    m = model.SimpleModel({
        'vocab_size': 5,
        'hidden_size': 64,
        'max_seq_len': 50,
        'n_heads': 4,
        'causal': False,
        'output_size': 2,
    })
    x = torch.ones((5, 10), dtype=torch.int)
    y = m(x)
    assert tuple(y.shape) == (5, 10, 2)


def test_wordpiece():
    word_counts = {
        'hello': 3,
        'what': 1,
        'man': 2,
        'bruh': 9,
        'mane': 3,
    }
    wp = wordpiece.WordPiece(16)
    wp.fit(word_counts)
    assert len(wp.vocab) == 16

    assert wp.tokenize(['hello', 'man', 'lol']) == ['he', '##l', '##l', '##o', 'm', '##a', '##n', '[UNK]']
