from torch import nn
import torch.nn.init as init

from nodules.attention import RoPEMultiHeadAttention
from nodules.rmsnorm import RMSNorm
from nodules.swiglu import SwiGLU


class SimpleModel(nn.Module):
    # TODO will this work? IDK
    def __init__(self, config):
        super().__init__()

        self.config = config  # for serialization I guess

        self.embedding = nn.Embedding(config['vocab_size'], config['hidden_size'])

        self.rms = RMSNorm(config['hidden_size'], eps=1e-8)
        self.rope_attention = RoPEMultiHeadAttention(config['hidden_size'], config['max_seq_len'],
                                                     config['n_heads'], config['causal'])
        self.linear = nn.Sequential(
            nn.Linear(config['hidden_size'], config['hidden_size']),
            SwiGLU(config['hidden_size'])
        )

        self.last_linear = nn.Linear(config['hidden_size'], config['output_size'])

        print('Model params:', sum([m.numel() for m in self.parameters()]))

    def forward(self, x):
        """
        Forward
        :param x: input, expected B x seq_len
        :return: output, expected B x seq_len x output_size
        """
        x = self.embedding(x)

        x = self.rms(x)
        x = x + self.rope_attention(x)

        x = self.rms(x)
        x = x + self.linear(x)

        logits = self.last_linear(x)

        return logits


def init_params(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)
