import torch
from torch import nn


class SwiGLU(nn.Module):
    """
    Swish_{beta}(linear_gate(x)) * (linear(x))
    """

    def __init__(self, size):
        super().__init__()
        self.linear_gate = nn.Linear(size, size)
        self.linear = nn.Linear(size, size)

        self.beta = nn.Parameter(torch.ones(1))
        self.register_parameter('beta', self.beta)

    def forward(self, x):
        gated = self.linear_gate(x)
        swish_gate = gated * torch.sigmoid(self.beta * gated)

        out = swish_gate * self.linear(x)
        return out
