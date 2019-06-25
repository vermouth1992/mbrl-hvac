import numpy as np
import torch

from torchlib.common import convert_numpy_to_tensor, FloatTensor
from torchlib.utils.random.sampler import BaseSampler, IntSampler
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

    def predict(self, state, actions):
        states = np.expand_dims(state, axis=0)
        actions = self.action_sampler.sample((self.horizon, self.num_random_action_selection))
        states = np.tile(states, (self.num_random_action_selection, 1))
        states = convert_numpy_to_tensor(states)
        actions = convert_numpy_to_tensor(actions)

        with torch.no_grad():
            cost = torch.zeros(size=(self.num_random_action_selection,)).type(FloatTensor)
            for i in range(self.horizon):
                next_states = self.model.predict_next_states(states, actions[i])
                cost += self.cost_fn(states, actions[i], next_states)
                states = next_states

            best_action = actions[0, torch.argmin(cost, dim=0)]
            best_action = best_action.cpu().numpy()
            return best_action