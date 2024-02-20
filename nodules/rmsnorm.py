from torch import nn
import torch


class RMSNorm(nn.Module):
    """
    Calculates the Frobenius norm of each input tensor and divides inputs by that norm.

    Then multiplies by a scaling value. Note that there's one scaling value
    for each output
    """

    def __init__(self, hidden_size, eps=1e-8):
        super().__init__()

        self.eps = eps
        self.scale = nn.Parameter(torch.ones(hidden_size))
        self.register_parameter('scale', self.scale)

    def forward(self, x):
        """
        Forward Pass.

        :param x: inputs, assumed shape is batch_size x seq_len x hidden_size
        :return: bro idk like the output??
        """
        frob_norm = torch.linalg.norm(x, dim=(1, 2), keepdim=True)  # sqrt(sum xi^2)
        rms = frob_norm * (x[0].numel() ** -0.5)  # divide by sqrt(n) to get RMS

        normalized = x / (rms + self.eps)
        return self.scale * normalized
