"""
Predictive model for model learning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchlib.common import move_tensor_to_gpu
from torchlib.deep_rl.algorithm.model_based import DeterministicWorldModel
from torchlib.deep_rl.models.policy import BasePolicy, _BetaActionHead
from torchlib.utils.math import normalize, unnormalize
from tqdm.auto import tqdm


class LSTMAttention(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(LSTMAttention, self).__init__()
        self.discrete = False
        feature_dim = state_dim + action_dim
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_size, num_layers=1, batch_first=True,
                            bidirectional=False)
        self.attention_matrix = nn.Parameter(torch.randn(hidden_size, 1), requires_grad=True)
        self.dropout = nn.Dropout(0.1)
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

        h = self.dropout.forward(h)

        score = self.out_linear.forward(h)

        return score


class EnergyPlusDynamicsModel(DeterministicWorldModel):
    """
    The only difference is that energyplus dynamics model takes in historical states and actions.
    """

    def __init__(self, state_dim=6, action_dim=4, hidden_size=32, learning_rate=1e-3, log_dir=None):
        dynamics_model = LSTMAttention(state_dim=state_dim, action_dim=action_dim, hidden_size=hidden_size)
        optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=learning_rate)

        self.writer = SummaryWriter(log_dir=log_dir + '/dynamics')
        self.global_step = 0

        super(EnergyPlusDynamicsModel, self).__init__(dynamics_model=dynamics_model, optimizer=optimizer)

    def predict_normalized_delta_next_state(self, states, actions):
        states_normalized = normalize(states, self.state_mean, self.state_std)
        if not self.dynamics_model.discrete:
            actions = normalize(actions, self.action_mean, self.action_std)
        predicted_delta_state_normalized = self.dynamics_model.forward(states_normalized, actions)
        return predicted_delta_state_normalized

    def fit_dynamic_model(self, dataset, epoch=10, batch_size=128, verbose=False):
        t = range(epoch)
        if verbose:
            t = tqdm(t)

        train_data_loader, val_data_loader = dataset.random_iterator(batch_size=batch_size)

        for i in t:
            losses = []
            for states, actions, next_states, _, _ in train_data_loader:
                # convert to tensor
                states = move_tensor_to_gpu(states)
                actions = move_tensor_to_gpu(actions)
                next_states = move_tensor_to_gpu(next_states)
                delta_states = next_states - states[:, -1, :]
                # calculate loss
                self.optimizer.zero_grad()
                predicted_delta_state_normalized = self.predict_normalized_delta_next_state(states, actions)
                delta_states_normalized = normalize(delta_states, self.delta_state_mean, self.delta_state_std)
                loss = F.mse_loss(predicted_delta_state_normalized, delta_states_normalized)

                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

            self.eval()
            val_losses = []
            with torch.no_grad():
                for states, actions, next_states, _, _ in val_data_loader:
                    # convert to tensor
                    states = move_tensor_to_gpu(states)
                    actions = move_tensor_to_gpu(actions)
                    next_states = move_tensor_to_gpu(next_states)
                    delta_states = next_states - states[:, -1, :]
                    predicted_delta_state_normalized = self.predict_normalized_delta_next_state(states, actions)
                    delta_states_normalized = normalize(delta_states, self.delta_state_mean, self.delta_state_std)
                    loss = F.mse_loss(predicted_delta_state_normalized, delta_states_normalized)
                    val_losses.append(loss.item())
            self.train()

            train_loss = np.mean(losses)
            val_loss = np.mean(val_losses)

            if verbose:
                t.set_description('Epoch {}/{} - Avg model train loss: {:.4f} - Avg model val loss: {:.4f}'.format(
                    i + 1, epoch, train_loss, val_loss))

            self.writer.add_scalars('dynamics/train_loss', {'train_loss': train_loss,
                                                            'val_loss': val_loss}, self.global_step)

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


class LSTMOutput(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(LSTMOutput, self).__init__()
        self.lstm = nn.LSTM(input_size=state_dim + action_dim, hidden_size=hidden_size, num_layers=1,
                            batch_first=True, bidirectional=False)

    def forward(self, state):
        out = self.lstm.forward(state)[0][:, -1, :]
        return out


class EnergyPlusPPOContinuousPolicy(BasePolicy):
    def __init__(self, state_dim, action_dim, hidden_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        super(EnergyPlusPPOContinuousPolicy, self).__init__(recurrent=False, hidden_size=None)

    def _create_feature_extractor(self):
        return LSTMOutput(self.state_dim, self.action_dim, self.hidden_size)

    def _create_action_head(self, feature_output_size):
        action_header = _BetaActionHead(feature_output_size, self.action_dim)
        return action_header

    def _calculate_feature_output_size(self):
        return self.hidden_size
