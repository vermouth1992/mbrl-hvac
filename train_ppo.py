"""
Test Vanilla PG on standard environment, where state is (ob_dim) and action is continuous/discrete
"""

import os

os.environ['CUDA'] = 'False'

import pprint

import gym.spaces
import numpy as np
import torch.optim
from torchlib import deep_rl
from agent.model import EnergyPlusPPOContinuousPolicy
from torchlib.utils.random import set_global_seeds

from gym_energyplus import make_env, ALL_CITIES


def make_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--discount', type=float, default=0.95)
    parser.add_argument('--gae_lambda', type=float, default=0.98)
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--value_coef', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=96 * 73)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--nn_size', '-s', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--city', type=str, choices=ALL_CITIES, nargs='+')
    parser.add_argument('--temp_center', type=float, default=23.5)
    parser.add_argument('--temp_tolerance', type=float, default=1.5)
    parser.add_argument('--window_length', type=int, default=20)
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    pprint.pprint(vars(args))

    set_global_seeds(args.seed)

    city = args.city
    temperature_center = args.temp_center
    temperature_tolerance = args.temp_tolerance

    log_dir = 'runs/{}_{}_{}_{}_ppo'.format('_'.join(city), temperature_center, args.discount, temperature_tolerance)

    env = make_env(city, temperature_center, temperature_tolerance, obs_normalize=True,
                   num_days_per_episode=1, window_length=args.window_length, log_dir=log_dir)

    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    if not discrete:
        print('Action space high', env.action_space.high)
        print('Action space low', env.action_space.low)

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    if discrete:
        policy_net = None
    else:
        policy_net = EnergyPlusPPOContinuousPolicy(state_dim=ob_dim, action_dim=ac_dim, hidden_size=args.nn_size)

    policy_optimizer = torch.optim.Adam(policy_net.parameters(), args.learning_rate)

    agent = deep_rl.algorithm.ppo.PPOAgent(policy_net, policy_optimizer,
                                           init_hidden_unit=None,
                                           lam=args.gae_lambda,
                                           clip_param=args.clip_param,
                                           entropy_coef=args.entropy_coef, value_coef=args.value_coef)

    checkpoint_path = 'checkpoint/{}_{}_{}_{}_ppo.ckpt'.format(city, temperature_center,
                                                               args.discount, temperature_tolerance)

    agent.train(args.exp_name, env, args.n_iter, args.discount, args.batch_size, np.inf,
                logdir=None, seed=args.seed, checkpoint_path=None)
