"""
Pytorch implementation of Vanilla PG. The code structure is adapted from UC Berkeley CS294-112.
Use various optimization techniques
1. Reward-to-go
2. Neural network baseline
3. Normalize advantage
4. Multiple threads to sample trajectory.
5. GAE-lambda
6. Multiple step update for PG
"""

import datetime
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from torchlib.common import FloatTensor, enable_cuda, convert_numpy_to_tensor
from torchlib.deep_rl import BaseAgent
from .utils import compute_reward_to_go_gae
from .utils import sample_trajectories, pathlength


class A2CAgent(BaseAgent):
    def __init__(self, policy_net: nn.Module, policy_optimizer, init_hidden_unit=None, nn_baseline=True,
                 lam=None, value_coef=0.5, max_grad_norm=0.5, initial_state_mean=0., initial_state_std=0.):
        super(A2CAgent, self).__init__()
        self.policy_net = policy_net
        if enable_cuda:
            self.policy_net.cuda()
        self.policy_optimizer = policy_optimizer
        self.nn_baseline = nn_baseline
        self.baseline_loss = None if not nn_baseline else nn.MSELoss()
        self.lam = lam
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.recurrent = init_hidden_unit is not None

        self.state_value_mean = initial_state_mean
        self.state_value_std = initial_state_std

        if init_hidden_unit is not None:
            self.init_hidden_unit = init_hidden_unit.astype(np.float32)
        else:
            self.init_hidden_unit = np.zeros(shape=(1), dtype=np.float32)  # dummy hidden unit for feed-forward policy

        assert self.init_hidden_unit.dtype == np.float32, 'hidden data type must be float32'

        assert isinstance(self.init_hidden_unit, np.ndarray), 'Type of init_hidden_unit {}'.format(
            type(init_hidden_unit))
        assert len(self.init_hidden_unit.shape) == 1

    def reset(self):
        self.hidden_unit = self.init_hidden_unit.copy()

    def get_hidden_unit(self):
        return self.hidden_unit

    def predict(self, state):
        """ Run the forward path of policy_network without gradient.

        Args:
            state: (batch_size, ob_dim)
            if discrete: probability distribution of a categorical distribution over actions
                sy_logits_na: (batch_size, self.ac_dim)
            if continuous: (mean, log_std) of a Gaussian distribution over actions
                sy_mean: (batch_size, self.ac_dim)
                sy_logstd: (self.ac_dim,)

        Returns:
            sy_sampled_ac:
                if discrete: (batch_size,)
                if continuous: (batch_size, self.ac_dim)

        """
        state = np.expand_dims(state, axis=0)
        self.hidden_unit = np.expand_dims(self.hidden_unit, axis=0)
        with torch.no_grad():
            state = torch.from_numpy(state).type(FloatTensor)
            hidden = torch.from_numpy(self.hidden_unit).type(FloatTensor)
            action_dist, hidden, _ = self.policy_net.forward(state, hidden)
            self.hidden_unit = hidden.cpu().numpy()[0]
            action = action_dist.sample(torch.Size([])).cpu().numpy()
            return action[0]

    def get_baseline_loss(self, raw_baseline, rewards):
        # update baseline
        # rewards = (rewards - torch.mean(rewards)) / (torch.std(rewards, unbiased=False) + eps)
        rewards = rewards.type(FloatTensor)
        loss = self.baseline_loss(raw_baseline, rewards)
        return loss

    def construct_dataset(self, paths, gamma):
        rewards, advantage, self.state_value_mean, self.state_value_std = compute_reward_to_go_gae(paths, gamma,
                                                                                                   self.policy_net,
                                                                                                   self.lam,
                                                                                                   self.state_value_mean,
                                                                                                   self.state_value_std)

        observation = np.concatenate([path["observation"] for path in paths])
        hidden = np.concatenate([path["hidden"] for path in paths])
        mask = np.concatenate([path["mask"] for path in paths])
        actions = np.concatenate([path['actions'] for path in paths])

        if not self.nn_baseline:
            advantage = rewards

        return actions, advantage, observation, rewards, hidden, mask

    def update_policy(self, dataset, epoch=1):
        """ Update policy

        Args:
            paths: a list of trajectories. Each contain a list of symbolic log_prob and rewards

        Returns:

        """
        actions, advantage, observation, rewards, hidden, mask = dataset

        observation = convert_numpy_to_tensor(observation)
        actions = convert_numpy_to_tensor(actions)
        advantage = convert_numpy_to_tensor(advantage)
        rewards = convert_numpy_to_tensor(rewards)

        for _ in range(epoch):
            # update policy network
            self.policy_optimizer.zero_grad()
            # compute log prob, assume observation is small.
            if not self.recurrent:
                distribution, _, raw_baselines = self.policy_net.forward(observation, None)
                log_prob = distribution.log_prob(actions)
            else:
                log_prob = []
                raw_baselines = []
                zero_index = np.where(mask == 0)[0] + 1
                zero_index = zero_index.tolist()
                zero_index.insert(0, 0)
                for i in range(len(zero_index) - 1):
                    start_index = zero_index[i]
                    end_index = zero_index[i + 1]
                    current_obs = observation[start_index:end_index]
                    current_actions = actions[start_index:end_index]
                    current_hidden = convert_numpy_to_tensor(np.expand_dims(self.init_hidden_unit, axis=0))
                    current_dist, _, current_baseline = self.policy_net.forward(current_obs, current_hidden)
                    log_prob.append(current_dist.log_prob(current_actions))
                    raw_baselines.append(current_baseline)

                log_prob = torch.cat(log_prob, dim=0)
                raw_baselines = torch.cat(raw_baselines, dim=0)

            assert log_prob.shape == advantage.shape, 'log_prob length {}, advantage length {}'.format(log_prob.shape,
                                                                                                       advantage.shape)

            action_loss = torch.mean(-log_prob * advantage)
            loss = action_loss

            if self.nn_baseline:
                value_loss = self.get_baseline_loss(raw_baselines, rewards)
                loss = loss + value_loss * self.value_coef

            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)

            loss.backward()
            self.policy_optimizer.step()

    @property
    def state_dict(self):
        return self.policy_net.state_dict()

    def load_state_dict(self, states):
        self.policy_net.load_state_dict(states)

    def save_checkpoint(self, checkpoint_path):
        torch.save(self.state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        self.load_state_dict(state_dict)

    def predict_state_value(self, state):
        """ compute the state value using nn baseline

        Args:
            state: (batch_size, ob_dim)

        Returns: (batch_size,)

        """
        if not self.nn_baseline:
            raise ValueError('Baseline function is not defined')
        else:
            state = np.expand_dims(state, axis=0)
            state = torch.from_numpy(state).type(FloatTensor)
            with torch.no_grad():
                return self.policy_net.forward(state)[1].cpu().numpy()[0]

    def train(self, exp, env, n_iter, gamma, min_timesteps_per_batch, max_path_length, logdir=None, seed=1996,
              checkpoint_path=None):
        # Set random seeds
        env.seed(seed)
        # Maximum length for episodes
        max_path_length = max_path_length or env.spec.max_episode_steps

        total_timesteps = 0

        if logdir:
            writer = SummaryWriter(log_dir=os.path.join(logdir, exp))
        else:
            writer = None

        best_avg_return = -np.inf

        start_time = time.time()

        for itr in range(n_iter):
            paths, timesteps_this_batch = sample_trajectories(self, env, min_timesteps_per_batch, max_path_length)

            total_timesteps += timesteps_this_batch

            datasets = self.construct_dataset(paths, gamma)
            self.update_policy(datasets)

            print('-----------------------------------------------------------------------------------')
            print('Iteration {}/{} - Number of paths {} - Timesteps this batch {} - Total timesteps {}'.format(
                itr + 1,
                n_iter,
                len(paths),
                timesteps_this_batch,
                total_timesteps))

            print('Reward-to-go mean: {:.4f}, std: {:.4f}'.format(self.state_value_mean, self.state_value_std))

            # logger
            returns = [np.sum(path["reward"]) for path in paths]
            ep_lengths = [pathlength(path) for path in paths]
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            max_return = np.max(returns)
            min_return = np.min(returns)

            if avg_return >= best_avg_return:
                best_avg_return = avg_return
                if checkpoint_path:
                    print('Saving checkpoint to {}'.format(checkpoint_path))
                    self.save_checkpoint(checkpoint_path=checkpoint_path)

            if writer:
                writer.add_scalars('data/return', {'avg': avg_return,
                                                   'std': std_return,
                                                   'max': max_return,
                                                   'min': min_return}, itr)
                writer.add_scalars('data/episode_length', {'avg': np.mean(ep_lengths),
                                                           'std': np.std(ep_lengths)}, itr)

            del datasets, paths

            time_elapse = datetime.timedelta(seconds=int(time.time() - start_time))

            print(
                'Return {:.2f}±{:.2f} - Return range [{:.2f}, {:.2f}] - Best Avg Return {:.2f} - Time elapsed {}'.format(
                    avg_return,
                    std_return,
                    min_return,
                    max_return,
                    best_avg_return,
                    time_elapse))


def make_default_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='a2c')
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.98)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=2e-3)
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--recurrent', '-re', action='store_true')
    parser.add_argument('--hidden_size', type=int, default=20)
    parser.add_argument('--nn_size', '-s', type=int, default=64)
    parser.add_argument('--initial_state_mean', type=float, default=0.)
    parser.add_argument('--initial_state_std', type=float, default=0.)
    parser.add_argument('--value_coef', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1)
    return parser


def get_policy_net(env, args):
    import gym
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    if not discrete:
        print('Action space high', env.action_space.high)
        print('Action space low', env.action_space.low)

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    recurrent = args['recurrent']
    hidden_size = args['hidden_size']

    from torchlib.deep_rl.models.policy import NormalNNPolicy, CategoricalNNPolicy, BetaNNPolicy
    from torchlib.deep_rl.envs import is_atari_env, is_ple_game

    if len(env.observation_space.shape) == 1:
        # low dimensional environment
        if discrete:
            policy_net = CategoricalNNPolicy(nn_size=args['nn_size'], state_dim=ob_dim, action_dim=ac_dim,
                                             recurrent=recurrent, hidden_size=hidden_size)
        else:
            policy_net = BetaNNPolicy(recurrent=recurrent, nn_size=args['nn_size'], state_dim=ob_dim,
                                      action_dim=ac_dim, hidden_size=hidden_size)

        if enable_cuda:
            policy_net.cuda()

        return policy_net

    elif is_atari_env(env.spec.id) or is_ple_game(env.spec.id):
        if env.observation_space.shape[:2] == (84, 84):
            frame_history_len = env.observation_space.shape[-1]

            from torchlib.deep_rl.models.policy import AtariPolicy
            policy_net = AtariPolicy(recurrent=recurrent, hidden_size=hidden_size,
                                     num_channel=frame_history_len, action_dim=env.action_space.n)

            if enable_cuda:
                policy_net.cuda()

            return policy_net
        else:
            raise ValueError('Not a typical env. Please define custom network')
    else:
        raise ValueError('Not a typical env. Please define custom network')
