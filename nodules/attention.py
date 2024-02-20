# ATTENTION IS ALL I NEED

import torch
from torch import nn
import torch.nn.functional as F
from rope import get_rotary_matrix


class RoPECausalAttention(nn.Module):
    def __init__(self, hidden_size, max_seq_len=1000):
        super().__init__()
        self.q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, hidden_size, bias=False)

        self.r = get_rotary_matrix(max_seq_len, hidden_size)
        self.register_buffer('tril', torch.tril(torch.ones(max_seq_len, max_seq_len)))

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
        attn = attn.masked_fill(self.tril[:s, :s] == 0, float('-inf'))  # apply causal mask
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        if return_attn_weights:
            return out, attn  # read the subsequent graph vertically
        else:
            return out
