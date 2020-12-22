"""
Models for model-based RL
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlib.common import enable_cuda, move_tensor_to_gpu, convert_numpy_to_tensor, FloatTensor
from torchlib.utils.layers import freeze, unfreeze
from torchlib.utils.math import normalize, unnormalize
from tqdm import tqdm

from .utils import EpisodicDataset as Dataset


class WorldModel(object):
    def __init__(self, dynamics_model: nn.Module, optimizer):
        self.dynamics_model = dynamics_model
        self.optimizer = optimizer

        if enable_cuda:
            self.dynamics_model.cuda()

    def train(self):
        unfreeze(self.dynamics_model)
        self.dynamics_model.train()

    def eval(self):
        freeze(self.dynamics_model)
        self.dynamics_model.eval()

    def set_statistics(self, dataset):
        self.state_mean = convert_numpy_to_tensor(dataset.state_mean).unsqueeze(dim=0)
        self.state_std = convert_numpy_to_tensor(dataset.state_std).unsqueeze(dim=0)
        if self.dynamics_model.discrete:
            self.action_mean = None
            self.action_std = None
        else:
            self.action_mean = convert_numpy_to_tensor(dataset.action_mean).unsqueeze(dim=0)
            self.action_std = convert_numpy_to_tensor(dataset.action_std).unsqueeze(dim=0)
        self.delta_state_mean = convert_numpy_to_tensor(dataset.delta_state_mean).unsqueeze(dim=0)
        self.delta_state_std = convert_numpy_to_tensor(dataset.delta_state_std).unsqueeze(dim=0)

    def fit_dynamic_model(self, dataset: Dataset, epoch=10, batch_size=128, verbose=False):
        raise NotImplementedError

    def predict_next_state(self, state, action):
        states = np.expand_dims(state, axis=0)
        actions = np.expand_dims(action, axis=0)
        states = convert_numpy_to_tensor(states)
        actions = convert_numpy_to_tensor(actions)
        with torch.no_grad():
            next_state = self.predict_next_states(states, actions).cpu().numpy()[0]
        return next_state

    def predict_next_states(self, states, actions):
        raise NotImplementedError

    @property
    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, states):
        raise NotImplementedError


class DeterministicWorldModel(WorldModel):
    """
    deterministic model following equation s_{t+1} = s_{t} + f(s_{t}, a_{t})
    """

    def __init__(self, dynamics_model: nn.Module, optimizer):
        super(DeterministicWorldModel, self).__init__(dynamics_model=dynamics_model, optimizer=optimizer)
        self.state_mean = None
        self.state_std = None
        self.action_mean = None
        self.action_std = None
        self.delta_state_mean = None
        self.delta_state_std = None

    def predict_normalized_delta_next_state(self, states, actions):
        states_normalized = normalize(states, self.state_mean, self.state_std)
        if not self.dynamics_model.discrete:
            actions = normalize(actions, self.action_mean, self.action_std)
        predicted_delta_state_normalized = self.dynamics_model.forward(states_normalized, actions)
        return predicted_delta_state_normalized

    def fit_dynamic_model(self, dataset: Dataset, epoch=10, batch_size=128, verbose=False):
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
                delta_states = next_states - states
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
                    delta_states = next_states - states
                    predicted_delta_state_normalized = self.predict_normalized_delta_next_state(states, actions)
                    delta_states_normalized = normalize(delta_states, self.delta_state_mean, self.delta_state_std)
                    loss = F.mse_loss(predicted_delta_state_normalized, delta_states_normalized)
                    val_losses.append(loss.item())
            self.train()

            if verbose:
                t.set_description('Epoch {}/{} - Avg model train loss: {:.4f} - Avg model val loss: {:.4f}'.format(
                    i + 1, epoch, np.mean(losses), np.mean(val_losses)))

    def predict_next_states(self, states, actions):
        assert self.state_mean is not None, 'Please set statistics before training for inference.'
        states_normalized = normalize(states, self.state_mean, self.state_std)

        if not self.dynamics_model.discrete:
            actions = normalize(actions, self.action_mean, self.action_std)

        predicted_delta_state_normalized = self.dynamics_model.forward(states_normalized, actions)
        predicted_delta_state = unnormalize(predicted_delta_state_normalized, self.delta_state_mean,
                                            self.delta_state_std)
        return states + predicted_delta_state

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


class StochasticVariationalWorldModel(WorldModel):
    """
    Stochastic variational inference model. Using s_{t+1} to reconstruct s_{t+1} with a_{t} and
    s_{t} as conditional input. To use the model, output s_{t+1} with s_{t} and a_{t} and sampled
    latent variable z_{t}.
    We optimize three losses here:
    1) Prediction loss
    2) KL-divergence loss
    """

    def __init__(self, dynamics_model: nn.Module, inference_network: nn.Module,
                 optimizer, code_size, kl_loss_weight=1e-3, ):
        """

        Args:
            dynamics_model: conditional generator model
            inference_network: conditional encoder model. Output a distribution
            optimizer: optimizer
            code_size:
        """
        super(StochasticVariationalWorldModel, self).__init__(dynamics_model=dynamics_model,
                                                              optimizer=optimizer)
        self.inference_network = inference_network
        self.code_size = code_size

        if enable_cuda:
            self.inference_network.cuda()

    def train(self):
        super(StochasticVariationalWorldModel, self).train()
        self.inference_network.train()

    def eval(self):
        super(StochasticVariationalWorldModel, self).eval()
        self.inference_network.eval()

    def fit_dynamic_model(self, dataset: Dataset, epoch=10, batch_size=128, verbose=False):
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

                latent_distribution = self.inference_network.forward(next_states)

                z = latent_distribution.sample()

    def predict_next_states(self, states, actions, z=None):
        assert self.state_mean is not None, 'Please set statistics before training for inference.'
        states = normalize(states, self.state_mean, self.state_std)

        if not self.dynamics_model.discrete:
            actions = normalize(actions, self.action_mean, self.action_std)

        if z is None:
            z = self._sample_latent_code(states.shape[0])

        predicted_states_normalized = self.dynamics_model.forward(states, actions, z)
        predicted_states = unnormalize(predicted_states_normalized, self.state_mean, self.state_std)
        return predicted_states

    def state_dict(self):
        pass

    def load_state_dict(self, states):
        pass

    ### VAE helper methods

    def _sample_latent_code(self, batch_size):
        z = torch.randn(batch_size, self.code_size).type(FloatTensor)
        return z

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
