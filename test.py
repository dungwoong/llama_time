import torch
from nodules import attention, rmsnorm, rope


def test_attention_shape():
    x = torch.randn((64, 5, 32))  # B x S x H
    attn = attention.RoPECausalAttention(32, 10)
    y = attn(x)
    assert tuple(y.shape) == (64, 5, 32)


def test_rms_norm_shape():
    norm = rmsnorm.RMSNorm(32)
    x = torch.randn((64, 5, 32))
    y = norm(x)
    assert tuple(y.shape) == (64, 5, 32)
