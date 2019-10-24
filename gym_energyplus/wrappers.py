"""
Wrappers for energyplus env.
1. Combine system timesteps into control timesteps. We compare whether ob[:3] is identical for
current step and previous step. We use the same action within the same control step (15 min).
"""

import os
import shutil
from collections import deque

import gym.spaces as spaces
import numpy as np
import torch
from gym.core import Wrapper, ObservationWrapper, ActionWrapper
from torch.utils.tensorboard import SummaryWriter
from torchlib.common import convert_to_tensor
from torchlib.deep_rl.envs.model_based import ModelBasedEnv


class CostFnWrapper(Wrapper):
    def cost_fn(self, states, actions, next_states):
        return self.env.cost_fn(states, actions, next_states)


class RepeatAction(CostFnWrapper, ModelBasedEnv):
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


class EnergyPlusGradualActionWrapper(CostFnWrapper):
    def __init__(self, env, action_low, action_high, action_delta):
        super(EnergyPlusGradualActionWrapper, self).__init__(env=env)
        self.action_space = spaces.Box(low=-1., high=1., shape=self.env.action_space.low.shape, dtype=np.float32)
        self.observation_space = spaces.Box(low=np.concatenate((self.env.observation_space.low, self.action_space.low)),
                                            high=np.concatenate(
                                                (self.env.observation_space.high, self.action_space.high)),
                                            dtype=np.float32)
        self.action_low = action_low
        self.action_high = action_high
        self.action_delta = action_delta

    def reset(self, **kwargs):
        self.prev_action = (self.action_low + self.action_high) / 2.
        obs = self.env.reset(**kwargs)
        return self.observation(obs)

    def step(self, action: np.ndarray):
        self.prev_action = self.action(action)
        obs, reward, done, info = self.env.step(self.prev_action)
        obs = self.observation(obs)
        return obs, reward, done, info

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """ Concatenate prev_action and observation

        Args:
            observation:

        Returns:

        """
        normalized_prev_action = (2 * self.prev_action - (self.action_high + self.action_low)) \
                                 / (self.action_high - self.action_low)
        return np.concatenate((observation, normalized_prev_action), axis=0)

    def reverse_observation(self, normalized_obs):
        return normalized_obs[:self.env.observation_space.low.shape[0]]

    def reverse_observation_batch(self, normalized_obs):
        return normalized_obs[:, :self.env.observation_space.low.shape[0]]

    def reverse_observation_batch_tensor(self, normalized_obs):
        return normalized_obs[:, :self.env.observation_space.low.shape[0]]

    def action(self, action: np.ndarray) -> np.ndarray:
        assert self.action_space.contains(action), 'Action {} not in action space'.format(action)
        current_action = np.clip(self.prev_action + action * self.action_delta,
                                 a_min=self.action_low, a_max=self.action_high)
        return current_action

    def cost_fn(self, states, actions, next_states):
        states = self.reverse_observation_batch_tensor(states)
        next_states = self.reverse_observation_batch_tensor(next_states)
        return self.env.cost_fn(states, actions, next_states)


class EnergyPlusNormalizeActionWrapper(ActionWrapper):
    def __init__(self, env, action_low, action_high):
        super(EnergyPlusNormalizeActionWrapper, self).__init__(env=env)
        self.action_space = spaces.Box(low=-1., high=1., shape=self.env.action_space.low.shape)
        self.action_low = action_low
        self.action_high = action_high

    def action(self, action):
        assert self.action_space.contains(action), 'Action {} is invalid'.format(action)
        action = (action + 1.) / 2. * (self.action_high - self.action_low) + self.action_low
        return action


class EnergyPlusDiscreteActionWrapper(ActionWrapper):
    pass


class EnergyPlusSplitEpisodeWrapper(CostFnWrapper):
    """
    Break a super long episode env into small length episodes. Used for PPO
    1. If the user calls reset, it will remain at the originally step.
    2. If the user reaches the maximum length, return done.
    3. If the user touches the true done, yield done.
    """

    def __init__(self, env, max_steps=96 * 5, window_length=None):
        """

        Args:
            env:
            max_steps:
            window_length: None or int. If None, we assume the agent deals with window to save memory.
                           Otherwise, the environment stores the
        """
        super(EnergyPlusSplitEpisodeWrapper, self).__init__(env=env)
        assert max_steps > 0, 'max_steps must be greater than zero. Got {}'.format(max_steps)
        self.max_steps = max_steps
        self.true_done = True
        self.last_obs = None

        self.window_length = window_length

        if self.window_length is not None:
            assert window_length < max_steps, 'window must be less than max_steps'
            self.prev_states = deque(maxlen=window_length)
            self.prev_actions = deque(maxlen=window_length - 1)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.window_length is None:
            self.last_obs = obs
        else:
            self.prev_states.append(obs)
            self.prev_actions.append(action)

        if done:
            self.true_done = True
            info['true_done'] = True
            return self.get_obs(), reward, done, info

        info['true_done'] = False
        self.current_steps += 1

        if self.current_steps == self.max_steps:
            return self.get_obs(), reward, True, info
        elif self.current_steps < self.max_steps:
            return self.get_obs(), reward, done, info
        else:
            raise ValueError('Please call reset before step.')

    def get_obs(self):
        if self.window_length is None:
            return self.last_obs
        else:
            states = np.array(self.prev_states)
            actions = list(self.prev_actions)
            actions.insert(0, np.zeros_like(self.action_space.low))
            actions = np.array(actions)
            return np.concatenate((states, actions), axis=-1)

    def reset(self, **kwargs):
        if self.true_done:
            if self.window_length is None:
                self.last_obs = self.env.reset(**kwargs)
            else:
                # fill the window_length
                self.last_obs = self.env.reset(**kwargs)
                for _ in range(self.window_length - 1):
                    last_action = self.action_space.sample()
                    self.prev_states.append(self.last_obs)
                    self.prev_actions.append(last_action)
                    self.last_obs, reward, done, info = self.env.step(last_action)
                self.prev_states.append(self.last_obs)
            self.true_done = False
        self.current_steps = 0
        return self.get_obs()


class Monitor(CostFnWrapper):
    def __init__(self, env, log_dir):
        super(Monitor, self).__init__(env=env)
        assert log_dir is not None, "log_dir can't be None"
        if os.path.isdir(log_dir):
            shutil.rmtree(log_dir)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0
        self.episode_index = 0

    def step(self, action: np.ndarray):
        obs, reward, done, info = self.env.step(action)
        self.dump_csv(obs, action, reward)
        self.dump_tensorboard(obs, action, reward)
        self.global_step += 1
        if done:
            self.logger.close()
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.logger = open(os.path.join(self.log_dir, 'episode-{}.csv'.format(self.episode_index)), 'w')
        self.episode_index += 1
        self.dump_csv_header()
        return self.env.reset(**kwargs)

    def dump_csv_header(self):
        self.logger.write('outside_temperature,west_temperature,east_temperature,ite_power,hvac_power,' +
                          'west_setpoint,east_setpoint,west_airflow,east_airflow,reward\n')

    def dump_csv(self, obs, original_action, reward):
        self.logger.write('{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.4f}\n'.format(
            obs[0], obs[1], obs[2], obs[4], obs[5], original_action[0], original_action[1],
            original_action[2], original_action[3], reward))

    def dump_tensorboard(self, obs, original_action, reward):
        self.writer.add_scalar('observation/outside_temperature', obs[0], self.global_step)
        self.writer.add_scalar('observation/west_temperature', obs[1], self.global_step)
        self.writer.add_scalar('observation/east_temperature', obs[2], self.global_step)
        self.writer.add_scalar('observation/ite_power (MW)', obs[4] / 1e6, self.global_step)
        self.writer.add_scalar('observation/hvac_power (MW)', obs[5] / 1e6, self.global_step)
        self.writer.add_scalar('action/west_setpoint', original_action[0], self.global_step)
        self.writer.add_scalar('action/east_setpoint', original_action[1], self.global_step)
        self.writer.add_scalar('action/west_airflow', original_action[2], self.global_step)
        self.writer.add_scalar('action/east_airflow', original_action[3], self.global_step)
        self.writer.add_scalar('data/reward', reward, self.global_step)


class EnergyPlusObsWrapper(ObservationWrapper, CostFnWrapper):
    def __init__(self, env, temperature_center):
        super(EnergyPlusObsWrapper, self).__init__(env=env)
        self.obs_mean = np.array([temperature_center, temperature_center, temperature_center, 1e5, 5000.],
                                 dtype=np.float32)
        self.obs_max = np.array([30., 30., 30., 1e5, 1e4], dtype=np.float32)
        self.obs_mean_tensor = convert_to_tensor(self.obs_mean).unsqueeze(dim=0)
        self.obs_max_tensor = convert_to_tensor(self.obs_max).unsqueeze(dim=0)

        self.observation_space = spaces.Box(low=np.array([-1., -1., -1., -10., -10.]),
                                            high=np.array([1., 1., 1., 10.0, 10.0]),
                                            dtype=np.float32)

    def reverse_observation(self, normalized_obs):
        obs = normalized_obs * self.obs_max + self.obs_mean
        total_power = obs[3] + obs[4]
        obs = np.insert(obs, 3, total_power)
        return obs

    def reverse_observation_batch(self, normalized_obs):
        assert isinstance(normalized_obs, np.ndarray), 'normalized_obs must be np.ndarray. Got {}'.format(
            type(normalized_obs))
        obs = normalized_obs * np.expand_dims(self.obs_max, axis=0) + np.expand_dims(self.obs_mean, axis=0)
        total_power = obs[:, 3:4] + obs[:, 4:5]
        obs = np.concatenate((obs[:, :3], total_power, obs[:, 3:]), axis=-1)
        return obs

    def observation(self, observation):
        temperature_obs = observation[0:3]
        power_obs = observation[4:]
        obs = np.concatenate((temperature_obs, power_obs))
        return (obs - self.obs_mean) / self.obs_max

    def reverse_observation_batch_tensor(self, normalized_obs):
        assert isinstance(normalized_obs, torch.Tensor)
        obs = normalized_obs * self.obs_max_tensor + self.obs_mean_tensor
        total_power = obs[:, 3:4] + obs[:, 4:5]
        obs = torch.cat((obs[:, :3], total_power, obs[:, 3:]), dim=-1)
        return obs

    def cost_fn(self, states, actions, next_states):
        states = self.reverse_observation_batch_tensor(states)
        next_states = self.reverse_observation_batch_tensor(next_states)
        return self.env.cost_fn(states, actions, next_states)
