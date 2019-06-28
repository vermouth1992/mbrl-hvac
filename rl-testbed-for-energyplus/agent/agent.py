from collections import deque

import numpy as np
import torch
from gym.spaces import Space
from torchlib.common import map_location
from torchlib.deep_rl import BaseAgent
from torchlib.deep_rl.model_based.model import Model

from .utils import EpisodicHistoryDataset as Dataset


class VanillaAgent(BaseAgent):
    def __init__(self, model: Model, planner, window_length: int, baseline_agent):
        self.model = model
        self.planner = planner
        self.history_states = deque(maxlen=window_length - 1)
        self.history_actions = deque(maxlen=window_length - 1)
        self.baseline_agent = baseline_agent

    def reset(self):
        """ Only reset on True done of one episode. """
        self.history_states.clear()

    def train(self):
        self.model.train()

    def test(self):
        self.model.test()

    def save_checkpoint(self, checkpoint_path):
        print('Saving checkpoint to {}'.format(checkpoint_path))
        torch.save(self.model.state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        states = torch.load(checkpoint_path, map_location=map_location)
        self.model.load_state_dict(states)

    def set_statistics(self, initial_dataset: Dataset):
        self.model.set_statistics(initial_dataset)

    def predict(self, state):
        self.test()
        if len(self.history_states) < self.history_states.maxlen:
            action = self.baseline_agent.predict(state)
        else:
            action = self.planner.predict(np.array(self.history_states), np.array(self.history_actions), state)
        self.history_states.append(state)
        self.history_actions.append(action)
        return action

    def fit_dynamic_model(self, dataset: Dataset, epoch=60, batch_size=128, verbose=False):
        self.train()
        self.model.fit_dynamic_model(dataset, epoch, batch_size, verbose)
