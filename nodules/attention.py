# ATTENTION IS ALL I NEED

import torch
from torch import nn
import torch.nn.functional as F
from nodules.rope import get_rotary_matrix


class RoPESelfAttention(nn.Module):
    """
    Causal attention with RoPE embeddings
    """
    def __init__(self, hidden_size, max_seq_len=1000):
        super().__init__()
        self.q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, hidden_size, bias=False)

        self.register_buffer('r', get_rotary_matrix(max_seq_len, hidden_size))

    def forward(self, x, r, tril, return_attn_weights=False):
        b, s, h = x.shape

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # rotate q and k
        qr = torch.bmm(q.transpose(0, 1), r[:s]).transpose(0, 1)
        kr = torch.bmm(k.transpose(0, 1), r[:s]).transpose(0, 1)

        # Q, K, V are B x S x H
        # each row in attn is the coefficients for the linear combination of V
        # thus, attn_{i, j} represents weight of jth input when calculating the ith output
        attn = qr @ kr.transpose(1, 2) * (h ** -0.5)  # B x S x S attention weights
        if tril is not None:
            attn = attn.masked_fill(tril[:s, :s] == 0, float('-inf'))  # apply causal mask
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        if return_attn_weights:
            return out, attn  # read the subsequent graph vertically
        else:
            return out


class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, max_seq_len, n_heads):
        super().__init__()

        self.heads = nn.ModuleList([
            RoPESelfAttention(hidden_size, max_seq_len) for _ in range(n_heads)
        ])

        self.linear = nn.Linear(n_heads * hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, r, tril):
        heads = [h(x, r, tril) for h in self.heads]
        x = torch.cat(heads, dim=-1)

        x = self.linear(x)
        x = self.dropout(x)
        return x
