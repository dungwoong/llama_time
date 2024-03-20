import torch
import numpy as np


def get_rotary_matrix(max_context_length, embedding_dim):
    """
    Gets rotary matrix(embedding_dim = hidden_size)

    so eg. if you have B x seq_len x hidden_size as inputs, and bmm with R,
    you have to first transpose(0, 1) to get seq_len x B x hidden_size
    R will be seq_len x hidden_size x hidden_size

    so the BMM will come out as seq_len x B x hidden_size
    """
    rope = torch.zeros((max_context_length, embedding_dim, embedding_dim), requires_grad=False)

    # Check original paper for variable meanings
    for m in range(max_context_length):
        for i in range(embedding_dim // 2):
            theta_i = 10000 ** (-2. * (i - 1) / embedding_dim)
            m_theta_i = m * theta_i
            rope[m, 2 * i, 2 * i] = np.cos(m_theta_i)
            rope[m, 2 * i, 2 * i + 1] = -np.sin(m_theta_i)
            rope[m, 2 * i + 1, 2 * i] = np.sin(m_theta_i)
            rope[m, 2 * i + 1, 2 * i + 1] = np.cos(m_theta_i)
    return rope