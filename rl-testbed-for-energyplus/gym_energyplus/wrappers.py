"""
Wrappers for energyplus env.
1. Combine system timesteps into control timesteps. We compare whether ob[:3] is identical for
current step and previous step. We use the same action within the same control step (15 min).
"""

import gym.spaces
import numpy as np
from gym.core import Wrapper
from torchlib.deep_rl.envs.model_based import ModelBasedEnv


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


class EnergyPlusWrapper(Wrapper, ModelBasedEnv):
    """
    Break a super long episode env into small length episodes. Used for PPO
    1. If the user calls reset, it will remain at the originally step.
    2. If the user reaches the maximum length, return done.
    3. If the user touches the true done, yield done.
    """

    def __init__(self, env, max_steps=96 * 5):
        super(EnergyPlusWrapper, self).__init__(env=env)
        assert max_steps > 0, 'max_steps must be greater than zero. Got {}'.format(max_steps)
        self.max_steps = max_steps
        self.true_done = True
        self.last_obs = None

        self.action_space = gym.spaces.Box(low=-1., high=1., shape=self.env.action_space.low.shape)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.last_obs = obs
        if done:
            self.true_done = True
            info['true_done'] = True
            return self.get_obs(), reward, done, info

        info['true_done'] = False
        self.current_steps += 1

        if self.current_steps == self.max_steps:
            return self.get_obs(), reward, True, info

        return self.get_obs(), reward, done, info

    def get_obs(self):
        return self.last_obs

    def cost_fn(self, states, actions, next_states):
        """ We want to minimize the total power consumption

        Args:
            states: current state (outside_temp, west_temp, east_temp, total_power, ite_power, hvac_power)
            actions: action (west_setpoint, east_setpoint, west_airflow, east_airflow)
            next_states: next state (outside_temp, west_temp, east_temp, total_power, ite_power, hvac_power)

        Returns: Combine total power consumption and constraints into a single metric

        """
        west_temperature = states[:, 1]
        east_temperature = states[:, 2]
        total_power_consumption = states[:, 3]

    def reset(self, **kwargs):
        if self.true_done:
            self.last_obs = self.env.reset(**kwargs)
            self.true_done = False
        self.current_steps = 0
        return self.get_obs()
