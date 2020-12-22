"""
Defines environment for model based RL
"""
from gym import Env


class ModelBasedEnv(Env):
    def cost_fn(self, states, actions, next_states):
        """ compute the cost of a transition of a batch of data.

        Args:
            states: (batch_size, ...)
            actions: (batch_size, ...)
            next_states: (batch_size, ...)

        Returns: (batch_size,)

        """
        raise NotImplementedError
