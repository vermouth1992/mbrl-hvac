"""
Common utilities to implement policy gradient algorithms
"""

import numpy as np
import torch
from scipy import signal

from torchlib.common import FloatTensor, convert_numpy_to_tensor, move_tensor_to_gpu, eps
from torchlib.dataset.utils import create_data_loader


def discount(x, gamma):
    return signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def pathlength(path):
    return len(path["reward"])


def compute_reward_to_go_gae(paths, gamma, policy_net, lam, value_mean, value_std):
    rewards = []
    gaes = []
    for path in paths:
        # compute last state value
        if path['mask'][-1] == 1:
            with torch.no_grad():
                last_obs = convert_numpy_to_tensor(
                    np.expand_dims(path['last_obs'], axis=0)).type(FloatTensor)
                last_hidden = convert_numpy_to_tensor(
                    np.expand_dims(path['last_hidden'], axis=0)).type(FloatTensor)
                last_state_value = policy_net.forward(last_obs, last_hidden)[-1].cpu().numpy()[0]
                last_state_value = last_state_value * value_std + value_mean
        else:
            last_state_value = 0.

        # we need to clip last_state_value by (max_abs_value / (1 - gamma))
        # Otherwise, large state value would cause positive feedback loop and cause the reward to explode.
        max_abs_value = np.max(np.abs(path['reward']))
        last_state_value = np.clip(last_state_value, a_min=-max_abs_value / (1 - gamma),
                                   a_max=max_abs_value / (1 - gamma))

        # calculate reward-to-go
        path['reward'].append(last_state_value)
        current_rewards = discount(path['reward'], gamma).astype(np.float32)

        rewards.append(current_rewards[:-1])

        # compute gae
        with torch.no_grad():
            observation = path['observation']
            hidden = path['hidden']
            data_loader = create_data_loader((observation, hidden), batch_size=32, shuffle=False, drop_last=False)
            values = []
            for obs, hid in data_loader:
                obs = move_tensor_to_gpu(obs)
                hid = move_tensor_to_gpu(hid)
                values.append(policy_net.forward(obs, hid)[-1])
            values = torch.cat(values, dim=0).cpu().numpy()
            values = values * value_std + value_mean
            values = np.append(values, last_state_value)

        # add the value of last obs for truncated trajectory
        temporal_difference = path['reward'][:-1] + values[1:] * gamma - values[:-1]
        # calculate reward-to-go
        gae = discount(temporal_difference, gamma * lam).astype(np.float32)
        gaes.append(gae)

    rewards = np.concatenate(rewards)
    new_values_mean, new_values_std = np.mean(rewards), np.std(rewards)
    rewards = (rewards - new_values_mean) / (new_values_std + eps)

    gaes = np.concatenate(gaes)
    gaes = (gaes - np.mean(gaes)) / (np.std(gaes) + eps)

    return rewards, gaes, new_values_mean, new_values_std


def sample_trajectory(agent, env, max_path_length):
    # this function should not participate in the computation graph
    ob = env.reset()
    agent.reset()
    actions, rewards, obs, hiddens, masks = [], [], [], [], []
    steps = 0
    while True:
        if ob.dtype == np.float:
            ob = ob.astype(np.float32)

        obs.append(ob)
        hiddens.append(agent.get_hidden_unit())

        ac = agent.predict(ob)

        if isinstance(ac, np.ndarray) and ac.dtype == np.float:
            ac = ac.astype(np.float32)

        actions.append(ac)

        ob, rew, done, _ = env.step(ac)
        rewards.append(rew)
        masks.append(int(not done))  # if done, mask is 0. Otherwise, 1.
        steps += 1
        if done or steps >= max_path_length:
            break

    if ob.dtype == np.float:
        ob = ob.astype(np.float32)

    path = {"actions": actions,
            "reward": rewards,
            "observation": np.array(obs),
            "hidden": np.array(hiddens),
            "mask": np.array(masks),
            'last_obs': ob,
            'last_hidden': agent.get_hidden_unit(),
            }
    return path


def sample_trajectories(agent, env, min_timesteps_per_batch, max_path_length):
    timesteps_this_batch = 0
    paths = []
    while True:
        path = sample_trajectory(agent, env, max_path_length)
        paths.append(path)
        timesteps_this_batch += pathlength(path)
        if timesteps_this_batch >= min_timesteps_per_batch:
            break
    return paths, timesteps_this_batch
