"""
Predictive model for model learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlib.common import convert_numpy_to_tensor
from torchlib.deep_rl.algorithm.model_based import DeterministicWorldModel
from torchlib.deep_rl.algorithm.model_based.utils import EpisodicDataset as Dataset
from torchlib.utils.math import normalize, unnormalize


class LSTMAttention(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(LSTMAttention, self).__init__()
        self.discrete = False
        feature_dim = state_dim + action_dim
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_size, num_layers=1, batch_first=True,
                            bidirectional=False)
        self.attention_matrix = nn.Parameter(torch.randn(hidden_size, 1), requires_grad=True)
        # self.dropout = nn.Dropout(0.5)
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


class EnergyPlusDynamicsModel(DeterministicWorldModel):
    """
    The only difference is that energyplus dynamics model takes in historical states and actions.
    """

    def __init__(self, state_dim=6, action_dim=4, hidden_size=32, learning_rate=1e-3):
        dynamics_model = LSTMAttention(state_dim=state_dim, action_dim=action_dim, hidden_size=hidden_size)
        optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=learning_rate)

        super(EnergyPlusDynamicsModel, self).__init__(dynamics_model=dynamics_model, optimizer=optimizer)

    def set_statistics(self, initial_dataset: Dataset):
        self.state_mean = convert_numpy_to_tensor(initial_dataset.state_mean).unsqueeze(dim=0)
        self.state_std = convert_numpy_to_tensor(initial_dataset.state_std).unsqueeze(dim=0)
        if self.dynamics_model.discrete:
            self.action_mean = None
            self.action_std = None
        else:
            self.action_mean = convert_numpy_to_tensor(initial_dataset.action_mean).unsqueeze(dim=0)
            self.action_std = convert_numpy_to_tensor(initial_dataset.action_std).unsqueeze(dim=0)
        self.delta_state_mean = convert_numpy_to_tensor(initial_dataset.delta_state_mean).unsqueeze(dim=0)
        self.delta_state_std = convert_numpy_to_tensor(initial_dataset.delta_state_std).unsqueeze(dim=0)

    def predict_next_states(self, states, actions):
        """

        Args:
            states: (batch_size, window_length, 6)
            actions: (batch_size, window_length, 4)

        Returns: next obs of shape (batch_size, 6)

        """
        assert self.state_mean is not None, 'Please set statistics before training for inference.'
        states_normalized = normalize(states, self.state_mean, self.state_std)

        if not self.dynamics_model.discrete:
            actions = normalize(actions, self.action_mean, self.action_std)

        predicted_delta_state_normalized = self.dynamics_model.forward(states_normalized, actions)
        predicted_delta_state = unnormalize(predicted_delta_state_normalized, self.delta_state_mean,
                                            self.delta_state_std)
        return states[:, -1, :] + predicted_delta_state
