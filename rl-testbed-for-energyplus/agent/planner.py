import numpy as np
import torch
from torchlib.common import convert_numpy_to_tensor, FloatTensor
from torchlib.utils.random.sampler import BaseSampler

from .model import Model


class Planner(object):
    """
    Planner predict the next best action given current state using model. Planner typically doesn't have memory.
    """

    def __init__(self, model: Model):
        self.model = model

    def predict(self, history_state, history_actions, current_state):
        raise NotImplementedError


class BestRandomActionPlanner(Planner):
    def __init__(self, model, action_sampler: BaseSampler, cost_fn=None,
                 horizon=15, num_random_action_selection=4096):
        """
        Args:
            model: Model instance. Can predict next states and optional cost (reward)
            action_sampler: Sampler that can sample actions
            cost_fn: if None, we expect model predicts both next states and cost (negative reward)
            horizon: Number of steps that we look into the future
            num_random_action_selection: Number of trajectories to sample
        """
        super(BestRandomActionPlanner, self).__init__(model=model)
        self.action_sampler = action_sampler
        self.horizon = horizon
        self.num_random_action_selection = num_random_action_selection
        if cost_fn is None:
            self.cost_fn = model.cost_fn
        else:
            self.cost_fn = cost_fn

    def predict(self, history_state, history_actions, current_state):
        """

        Args:
            history_state: (T - 1, 6)
            history_actions: (T - 1, 4)
            current_state: (6,)

        Returns: best action (4,)

        """
        states = np.expand_dims(history_state, axis=0)  # (1, T - 1, 6)
        states = np.tile(states, (self.num_random_action_selection, 1, 1))  # (N, T - 1, 6)
        states = convert_numpy_to_tensor(states)

        next_states = np.expand_dims(current_state, axis=0)  # (1, 6)
        next_states = np.tile(next_states, (self.num_random_action_selection, 1))  # (N, 6)
        next_states = convert_numpy_to_tensor(next_states)

        actions = self.action_sampler.sample((self.horizon, self.num_random_action_selection))  # (H, N, 4)
        actions = convert_numpy_to_tensor(actions)

        history_actions = np.expand_dims(history_actions, axis=0)  # (1, T - 1, 4)
        current_action = np.tile(history_actions, (self.num_random_action_selection, 1, 1))  # (N, T - 1, 4)
        current_action = convert_numpy_to_tensor(current_action)

        with torch.no_grad():
            cost = torch.zeros(size=(self.num_random_action_selection,)).type(FloatTensor)
            for i in range(self.horizon):
                states = torch.cat((states, torch.unsqueeze(next_states, dim=1)), dim=1)  # # (N, T, 6)
                current_action = torch.cat((current_action, torch.unsqueeze(actions[i], dim=1)), dim=1)  # (N, T, 4)
                next_states = self.model.predict_next_states(states, current_action)  # (N, 6)
                cost += self.cost_fn(states[:, -1, :], actions[i], next_states)
                current_action = current_action[:, 1:, :]  # (N, T - 1, 4)
                states = states[:, 1:, :]

            best_action = actions[0, torch.argmin(cost, dim=0)]
            best_action = best_action.cpu().numpy()
            return best_action
