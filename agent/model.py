"""
Predictive model for model learning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlib.common import enable_cuda, move_tensor_to_gpu, convert_numpy_to_tensor
from torchlib.deep_rl.model_based.model import Model
from torchlib.deep_rl.model_based.utils import EpisodicDataset as Dataset
from torchlib.utils import normalize, unnormalize
from tqdm.auto import tqdm


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


class EnergyPlusDynamicsModel(Model):
    def __init__(self, state_dim=6, action_dim=4, hidden_size=32, learning_rate=1e-3):
        self.state_mean = None
        self.state_std = None
        self.action_mean = None
        self.action_std = None
        self.delta_state_mean = None
        self.delta_state_std = None

        self.dynamics_model = LSTMAttention(state_dim=state_dim, action_dim=action_dim, hidden_size=hidden_size)
        self.optimizer = torch.optim.Adam(self.dynamics_model.parameters(), lr=learning_rate)

        if enable_cuda:
            self.dynamics_model.cuda()

    def train(self):
        self.dynamics_model.train()

    def test(self):
        self.dynamics_model.eval()

    def set_statistics(self, initial_dataset: Dataset):
        self.state_mean = convert_numpy_to_tensor(initial_dataset.state_mean)
        self.state_std = convert_numpy_to_tensor(initial_dataset.state_std)
        if self.dynamics_model.discrete:
            self.action_mean = None
            self.action_std = None
        else:
            self.action_mean = convert_numpy_to_tensor(initial_dataset.action_mean)
            self.action_std = convert_numpy_to_tensor(initial_dataset.action_std)
        self.delta_state_mean = convert_numpy_to_tensor(initial_dataset.delta_state_mean)
        self.delta_state_std = convert_numpy_to_tensor(initial_dataset.delta_state_std)

        self.state_mean = torch.unsqueeze(torch.unsqueeze(self.state_mean, dim=0), dim=0)
        self.state_std = torch.unsqueeze(torch.unsqueeze(self.state_std, dim=0), dim=0)
        self.action_mean = torch.unsqueeze(torch.unsqueeze(self.action_mean, dim=0), dim=0)
        self.action_std = torch.unsqueeze(torch.unsqueeze(self.action_std, dim=0), dim=0)
        self.delta_state_mean = torch.unsqueeze(self.delta_state_mean, dim=0)
        self.delta_state_std = torch.unsqueeze(self.delta_state_std, dim=0)

    def fit_dynamic_model(self, dataset: Dataset, epoch=10, batch_size=128, verbose=False):
        t = range(epoch)
        if verbose:
            t = tqdm(t)

        for i in t:
            losses = []
            for states, actions, delta_states in dataset.random_iterator(batch_size=batch_size):
                # convert to tensor
                states = move_tensor_to_gpu(states)
                actions = move_tensor_to_gpu(actions)
                delta_states = move_tensor_to_gpu(delta_states)
                # calculate loss
                self.optimizer.zero_grad()
                states_normalized = normalize(states, self.state_mean, self.state_std)
                if not self.dynamics_model.discrete:
                    actions = normalize(actions, self.action_mean, self.action_std)
                delta_states_normalized = normalize(delta_states, self.delta_state_mean, self.delta_state_std)
                predicted_delta_state_normalized = self.dynamics_model.forward(states_normalized, actions)
                loss = F.mse_loss(predicted_delta_state_normalized, delta_states_normalized)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            if verbose:
                t.set_description('Epoch {}/{} - Avg model loss: {:.4f}'.format(i + 1, epoch, np.mean(losses)))

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

    def state_dict(self):
        states = {
            'dynamic_model': self.dynamics_model.state_dict(),
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'action_mean': self.action_mean,
            'action_std': self.action_std,
            'delta_state_mean': self.delta_state_mean,
            'delta_state_std': self.delta_state_std
        }
        return states

    def load_state_dict(self, states):
        self.dynamics_model.load_state_dict(states['dynamic_model'])
        self.state_mean = states['state_mean']
        self.state_std = states['state_std']
        self.action_mean = states['action_mean']
        self.action_std = states['action_std']
        self.delta_state_mean = states['delta_state_mean']
        self.delta_state_std = states['delta_state_std']
