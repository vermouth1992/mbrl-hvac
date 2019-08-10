from collections import deque

import numpy as np
from torchlib.deep_rl.algorithm.model_based import ModelBasedPlanAgent


class ModelBasedHistoryPlanAgent(ModelBasedPlanAgent):
    def __init__(self, model, planner, window_length: int, baseline_agent):
        super(ModelBasedHistoryPlanAgent, self).__init__(model=model, planner=planner)
        self.history_states = deque(maxlen=window_length - 1)
        self.history_actions = deque(maxlen=window_length - 1)
        self.baseline_agent = baseline_agent

    def reset(self):
        """ Only reset on True done of one episode. """
        self.history_states.clear()

    def predict(self, state):
        self.model.eval()
        if len(self.history_states) < self.history_states.maxlen:
            action = self.baseline_agent.predict(state)
        else:
            action = self.planner.predict(np.array(self.history_states), np.array(self.history_actions), state)
        self.history_states.append(state)
        self.history_actions.append(action)
        return action
