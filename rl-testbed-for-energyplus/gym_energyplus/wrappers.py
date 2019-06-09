"""
Wrappers for energyplus env.
1. Combine system timesteps into control timesteps. We compare whether ob[:3] is identical for
current step and previous step. We use the same action within the same control step (15 min).
"""

import numpy as np
from gym.core import Wrapper


class RepeatAction(Wrapper):
    def __init__(self, env):
        super(RepeatAction, self).__init__(env=env)
        self.last_obs = None
        self.reward = []
        self.obs = []

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.obs.append(obs)
        self.reward.append(reward)

        # repeat the same action until it is done or the obs is different.
        while np.array_equal(obs[:3], self.last_obs[:3]) and (not done):
            obs, reward, done, info = self.env.step(action)
            self.obs.append(obs)
            self.reward.append(reward)

        self.last_obs = obs

        obs = np.mean(self.obs, axis=0)
        reward = np.mean(self.reward)

        self.obs = []
        self.reward = []

        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_obs = obs
        return obs
