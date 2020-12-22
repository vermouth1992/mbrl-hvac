"""
Virtual Environment for training policy using model-free approach
"""

import random

import gym

from torchlib.deep_rl.envs.model_based import ModelBasedEnv
from torchlib.utils.random import set_global_seeds
from .world_model import WorldModel


class VirtualEnv(gym.Env):
    def __init__(self, model: WorldModel, real_env: ModelBasedEnv):
        """ Virtual Environment. We only consider environment with pre-defined cost function.
        We will extend to learn reward function in the end.

        Args:
            model: model to predict next states given previous states and
            real_env: real environment
        """
        super(VirtualEnv, self).__init__()
        self.action_space = real_env.action_space
        self.observation_space = real_env.observation_space
        self.reward_range = real_env.reward_range
        self.cost_fn = real_env.cost_fn
        self.model = model
        self.current_state = None

        self.initial_states_pool = None

        if real_env.spec is not None and hasattr(real_env.spec, 'max_episode_steps'):
            self.max_episode_steps = real_env.spec.max_episode_steps
        else:
            self.max_episode_steps = float('inf')

    def set_initial_states_pool(self, initial_states_pool):
        self.initial_states_pool = initial_states_pool

    def reset(self):
        self.current_steps = 0
        self.current_state = random.choice(self.initial_states_pool)
        return self.current_state

    def step(self, action):
        self.current_steps += 1
        next_state = self.model.predict_next_state(self.current_state, action)
        reward = -self.cost_fn(self.current_state, action, next_state)
        self.current_state = next_state

        if self.current_steps >= self.max_episode_steps:
            done = True
        else:
            done = False

        return self.current_state, reward, done, {}

    def seed(self, seed=None):
        if seed is not None:
            set_global_seeds(seed)
