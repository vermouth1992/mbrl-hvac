"""
Define model based wrapper for other well written environments.
Use to define cost_fn for state transition (state, action, next_state)
without learning reward functions.
"""

import gym
import numpy as np
import torch

from torchlib.common import LongTensor, FloatTensor


class ModelBasedWrapper(gym.Wrapper):
    def cost_fn_numpy_batch(self, states, actions, next_states):
        raise NotImplementedError

    def cost_fn_torch_batch(self, states, actions, next_states):
        raise NotImplementedError

    def cost_fn(self, state, action, next_state):
        if isinstance(state, torch.Tensor):
            states = torch.unsqueeze(state, dim=0)
            actions = torch.unsqueeze(action, dim=0)
            next_states = torch.unsqueeze(next_state, dim=0)
            return self.cost_fn_torch_batch(states, actions, next_states)[0]
        elif isinstance(state, np.ndarray):
            states = np.expand_dims(state, axis=0)
            actions = np.expand_dims(action, axis=0)
            next_states = np.expand_dims(next_state, axis=0)
            return self.cost_fn_numpy_batch(states, actions, next_states)[0]

        else:
            raise ValueError('Unknown data type {}'.format(type(state)))

    def cost_fn_batch(self, states, actions, next_states):
        if isinstance(states, torch.Tensor):
            return self.cost_fn_torch_batch(states, actions, next_states)

        elif isinstance(states, np.ndarray):
            return self.cost_fn_numpy_batch(states, actions, next_states)

        else:
            raise ValueError('Unknown data type {}'.format(type(states)))


class ModelBasedCartPoleWrapper(ModelBasedWrapper):
    version = 'v2'

    def cost_fn_numpy_batch_v1(self, states, actions, next_states):
        x = next_states[:, 0]
        theta = next_states[:, 2]
        x_done = np.logical_or(x < -self.x_threshold, x > self.x_threshold)
        theta_done = np.logical_or(theta < -self.theta_threshold_radians, theta > self.theta_threshold_radians)
        done = np.logical_or(x_done, theta_done).astype(np.int)
        return done

    def cost_fn_numpy_batch_v2(self, states, actions, next_states):
        x = next_states[:, 0]
        theta = next_states[:, 2]
        return np.abs(x) / self.x_threshold + np.abs(theta) / self.theta_threshold_radians

    def cost_fn_numpy_batch(self, states, actions, next_states):
        if self.version == 'v1':
            return self.cost_fn_numpy_batch_v1(states, actions, next_states)
        elif self.version == 'v2':
            return self.cost_fn_numpy_batch_v2(states, actions, next_states)
        else:
            raise NotImplementedError

    def cost_fn_torch_batch_v1(self, states, actions, next_states):
        x = next_states[:, 0]
        theta = next_states[:, 2]
        x_done = (x < -self.x_threshold) | (x > self.x_threshold)
        theta_done = (theta < -self.theta_threshold_radians) | (theta > self.theta_threshold_radians)
        done = x_done | theta_done
        done = done.type(LongTensor)
        return done

    def cost_fn_torch_batch_v2(self, states, actions, next_states):
        x = next_states[:, 0]
        theta = next_states[:, 2]
        return torch.abs(x) / self.x_threshold + torch.abs(theta) / self.theta_threshold_radians

    def cost_fn_torch_batch(self, states, actions, next_states):
        if self.version == 'v1':
            return self.cost_fn_torch_batch_v1(states, actions, next_states)
        elif self.version == 'v2':
            return self.cost_fn_torch_batch_v2(states, actions, next_states)
        else:
            raise NotImplementedError


class ModelBasedPendulumWrapper(ModelBasedWrapper):
    def cost_fn_numpy_batch(self, states, actions, next_states):
        cos_th, sin_th, thdot = states[:, 0], states[:, 1], states[:, 2]
        th = np.arctan2(sin_th, cos_th)

        costs = th ** 2 + .1 * thdot ** 2 + .001 * (actions[:, 0] ** 2)
        return costs

    def cost_fn_torch_batch(self, states, actions, next_states):
        cos_th, sin_th, thdot = states[:, 0], states[:, 1], states[:, 2]
        th = torch.atan2(sin_th, cos_th)
        costs = th ** 2 + .1 * thdot ** 2 + .001 * (actions[:, 0] ** 2)
        return costs


class ModelBasedRoboschoolInvertedPendulumWrapper(ModelBasedWrapper):
    def cost_fn_torch_batch(self, states, actions, next_states):
        cos_th, sin_th = next_states[:, 2], next_states[:, 3]
        theta = torch.atan2(sin_th, cos_th)
        cost = torch.abs(theta).type(FloatTensor)
        return cost

    def cost_fn_numpy_batch(self, states, actions, next_states):
        cos_th, sin_th = next_states[:, 2], next_states[:, 3]
        theta = np.arctan2(sin_th, cos_th)
        cost = np.abs(theta).astype(np.float32)
        return cost


class ModelBasedRoboschoolInvertedPendulumSwingupWrapper(ModelBasedWrapper):
    def cost_fn_numpy_batch(self, states, actions, next_states):
        return -next_states[:, 2]

    def cost_fn_torch_batch(self, states, actions, next_states):
        return -next_states[:, 2]


class ModelBasedRoboschoolReacher(ModelBasedWrapper):
    def cost_fn_numpy_batch(self, states, actions, next_states):
        old_to_target_vec = states[:, 2:4]
        to_target_vec = next_states[:, 2:4]
        theta_dot = next_states[:, 6]
        gamma = next_states[:, 7]
        gamma_dot = next_states[:, 8]

        old_potential = 100 * np.sqrt(np.sum(old_to_target_vec ** 2, axis=-1))
        potential = 100 * np.sqrt(np.sum(to_target_vec ** 2, axis=-1))

        electricity_cost = (
                0.10 * (np.abs(actions[:, 0] * theta_dot) + np.abs(actions[:, 1] * gamma_dot))
                + 0.01 * (np.abs(actions[:, 0]) + np.abs(actions[:, 1]))
        )

        stuck_joint_cost = 0.1 * (np.abs(np.abs(gamma) - 1) < 0.01).astype(np.float32)

        return potential - old_potential + electricity_cost + stuck_joint_cost

    def cost_fn_torch_batch(self, states, actions, next_states):
        old_to_target_vec = states[:, 2:4]
        to_target_vec = next_states[:, 2:4]
        theta_dot = next_states[:, 6]
        gamma = next_states[:, 7]
        gamma_dot = next_states[:, 8]

        old_potential = 100 * torch.sqrt(torch.sum(old_to_target_vec ** 2, dim=-1))
        potential = 100 * torch.sqrt(torch.sum(to_target_vec ** 2, dim=-1))

        electricity_cost = (
                0.10 * (torch.abs(actions[:, 0] * theta_dot) + torch.abs(actions[:, 1] * gamma_dot))
                + 0.01 * (torch.abs(actions[:, 0]) + torch.abs(actions[:, 1]))
        )

        stuck_joint_cost = 0.1 * (torch.abs(torch.abs(gamma) - 1) < 0.01).type(FloatTensor)

        return potential - old_potential + electricity_cost + stuck_joint_cost


model_based_wrapper_dict = {
    'CartPole-v0': ModelBasedCartPoleWrapper,
    'CartPole-v1': ModelBasedCartPoleWrapper,
    'CartPoleContinuous-v0': ModelBasedCartPoleWrapper,
    'CartPoleContinuous-v1': ModelBasedCartPoleWrapper,
    'Pendulum-v0': ModelBasedPendulumWrapper,
    'PendulumNormalized-v0': ModelBasedPendulumWrapper,
    'RoboschoolInvertedPendulum-v1': ModelBasedRoboschoolInvertedPendulumWrapper,
    'RoboschoolInvertedPendulumSwingup-v1': ModelBasedRoboschoolInvertedPendulumSwingupWrapper,
    'RoboschoolReacher-v1': ModelBasedRoboschoolReacher
}
