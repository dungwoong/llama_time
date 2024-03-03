import torch
from torch import nn
import torch.nn.functional as F

from nodules.rmsnorm import RMSNorm
from nodules.rope import get_rotary_matrix
from nodules.swiglu import SwiGLU


class LoraLinear(nn.Module):
    def __init__(self, in_features, out_features, middle_features):
        super().__init__()
        self.l1 = nn.Linear(in_features, middle_features, bias=False)
        self.l2 = nn.Linear(middle_features, out_features, bias=False)

    def forward(self, x):
        return self.l2(self.l1(x))


class RoPELORASelfAttention(nn.Module):
    """
    Causal attention with RoPE embeddings
    """

    def __init__(self, hidden_size, middle_features, max_seq_len=1000, causal=True):
        super().__init__()
        self.q = LoraLinear(hidden_size, hidden_size, middle_features)
        self.k = LoraLinear(hidden_size, hidden_size, middle_features)
        self.v = LoraLinear(hidden_size, hidden_size, middle_features)

        self.register_buffer('r', get_rotary_matrix(max_seq_len, hidden_size))
        self.causal = causal
        if causal:
            self.register_buffer('tril', torch.tril(torch.ones(max_seq_len, max_seq_len)))
        else:
            self.tril = None

    def forward(self, x, return_attn_weights=False):
        b, s, h = x.shape

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # rotate q and k
        qr = torch.bmm(q.transpose(0, 1), self.r[:s]).transpose(0, 1)
        kr = torch.bmm(k.transpose(0, 1), self.r[:s]).transpose(0, 1)

        # Q, K, V are B x S x H
        # each row in attn is the coefficients for the linear combination of V
        # thus, attn_{i, j} represents weight of jth input when calculating the ith output
        attn = qr @ kr.transpose(1, 2) * (h ** -0.5)  # B x S x S attention weights
        if self.causal:
            attn = attn.masked_fill(self.tril[:s, :s] == 0, float('-inf'))  # apply causal mask
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        if return_attn_weights:
            return out, attn  # read the subsequent graph vertically
        else:
            return out


class RoPELORAMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, max_seq_len, n_heads, middle_features, causal=True):
        super().__init__()

        self.heads = nn.ModuleList([
            RoPELORASelfAttention(hidden_size, middle_features, max_seq_len, causal) for _ in range(n_heads)
        ])

        self.linear = nn.Linear(n_heads * hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        heads = [h(x) for h in self.heads]
        x = torch.cat(heads, dim=-1)

        x = self.linear(x)
        x = self.dropout(x)
        return x


class LLaMABlock(nn.Module):
    """
    One Block of LLaMA
    """

    def __init__(self, config):
        super().__init__()

        self.rms = RMSNorm(config['hidden_size'], eps=1e-8)
        self.rope_attention = RoPELORAMultiHeadAttention(config['hidden_size'], config['max_seq_len'],
                                                         config['n_heads'], config['middle_features'], config['causal'])
        self.linear = nn.Sequential(
            nn.Linear(config['hidden_size'], config['hidden_size']),
            SwiGLU(config['hidden_size'])
        )

    def forward(self, x):
        # assume x is already B x seq_len x hidden_size
        x = self.rms(x)
        x = x + self.rope_attention(x)

        x = self.rms(x)
        x = x + self.linear(x)

        return x


class LORALLaMAModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        for k in ['vocab_size', 'hidden_size',
                  'max_seq_len', 'n_heads',
                  'causal', 'output_size',
                  'n_blocks', 'middle_features']:
            assert k in config
        self.config = config

        self.embedding = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.blocks = nn.Sequential(
            *[LLaMABlock(config) for _ in range(config['n_blocks'])]
        )

        self.rms = RMSNorm(config['hidden_size'], eps=1e-8)
        self.final_linear = nn.Linear(config['hidden_size'], config['vocab_size'])
        print('Model params:', sum([m.numel() for m in self.parameters()]))

    def forward(self, x):
        x = self.embedding(x)
        x = self.blocks(x)
        x = self.rms(x)
        x = self.final_linear(x)
        return x

