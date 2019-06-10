"""
Predictive model for model learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMAttention(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(LSTMAttention, self).__init__()
        feature_dim = state_dim + action_dim
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_size, num_layers=1, batch_first=True,
                            bidirectional=False)
        self.attention_matrix = nn.Parameter(torch.randn(hidden_size, 1), requires_grad=True)
        # self.dropout = nn.Dropout(0.5, inplace=True)
        self.out_linear = nn.Linear(hidden_size, state_dim)

    def forward(self, state, action):
        """ Compute forward pass of LSTM Attention layer by looking at history

        Args:
            state: (batch_size, T, 6)
            action: (batch_size, T, 4)

        Returns: delta_state: (batch_size, 6)

        """
        batch_size = state.size(0)

        feature = torch.cat((state, action), dim=-1)  # (b, T, 10)

        output, _ = self.lstm.forward(feature)  # (b, 10)

        M = torch.tanh(output)

        attention_matrix = self.attention_matrix.repeat(batch_size, 1, 1)

        alpha = F.softmax(torch.bmm(M, attention_matrix).squeeze(dim=-1), dim=-1)  # (batch_size, T)

        alpha = alpha.unsqueeze(dim=1)  # (batch_size, 1, T)

        r = torch.bmm(alpha, M).squeeze(dim=1)  # (batch_size, 1, hidden_size) -> (batch_size, hidden_size)

        h = torch.tanh(r)

        # h = self.dropout.forward(h)

        score = self.out_linear.forward(h)

        return score
