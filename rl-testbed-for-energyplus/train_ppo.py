"""
Test Vanilla PG on standard environment, where state is (ob_dim) and action is continuous/discrete
"""

import os

os.environ['CUDA'] = 'False'

import pprint

import gym.spaces
import numpy as np
import torch.optim
import torchlib.deep_rl.policy_gradient.ppo as ppo
from torchlib.deep_rl.models.policy import DiscreteNNPolicy, ContinuousNNPolicy
from torchlib.utils.random import set_global_seeds

from gym_energyplus.envs.energyplus_env import EnergyPlusEnv
from gym_energyplus.path import get_model_filepath, get_weather_filepath, energyplus_bin_path, ENERGYPLUS_WEATHER_dict
from gym_energyplus.wrappers import RepeatAction, EnergyPlusWrapper, Monitor, EnergyPlusObsWrapper


def make_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--discount', type=float, default=0.95)
    parser.add_argument('--gae_lambda', type=float, default=0.98)
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--value_coef', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=500)
    parser.add_argument('--batch_size', '-b', type=int, default=96 * 73)
    parser.add_argument('--recurrent', '-re', action='store_true')
    parser.add_argument('--hidden_size', type=int, default=20)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--nn_size', '-s', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--city', type=str, choices=ENERGYPLUS_WEATHER_dict.keys())
    parser.add_argument('--temp_center', type=float, default=23.5)
    parser.add_argument('--temp_tolerance', type=float, default=1.5)
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    pprint.pprint(vars(args))

    set_global_seeds(args.seed)

    city = args.city
    temperature_center = args.temp_center
    temp_tolerance = args.temp_tolerance
    num_days_per_episode = 1

    env = EnergyPlusEnv(energyplus_file=energyplus_bin_path,
                        model_file=get_model_filepath('temp_fan'),
                        weather_file=get_weather_filepath(city),
                        config={'temp_center': temperature_center, 'temp_tolerance': temp_tolerance},
                        log_dir=None,
                        verbose=True)
    env = RepeatAction(env)

    log_dir = 'runs/{}_{}_{}_{}_ppo'.format(city, temperature_center, args.discount, temp_tolerance)

    env = Monitor(env, log_dir=log_dir)
    env = EnergyPlusWrapper(env, max_steps=96 * num_days_per_episode)
    env = EnergyPlusObsWrapper(env)

    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    if not discrete:
        print('Action space high', env.action_space.high)
        print('Action space low', env.action_space.low)

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    recurrent = args.recurrent
    hidden_size = args.hidden_size

    if discrete:
        policy_net = DiscreteNNPolicy(nn_size=args.nn_size, state_dim=ob_dim, action_dim=ac_dim,
                                      recurrent=recurrent, hidden_size=hidden_size)
    else:
        policy_net = ContinuousNNPolicy(nn_size=args.nn_size, state_dim=ob_dim, action_dim=ac_dim,
                                        recurrent=recurrent, hidden_size=hidden_size)

    policy_optimizer = torch.optim.Adam(policy_net.parameters(), args.learning_rate)

    gae_lambda = args.gae_lambda

    if recurrent:
        init_hidden_unit = np.zeros(shape=(hidden_size))
    else:
        init_hidden_unit = None

    agent = ppo.Agent(policy_net, policy_optimizer,
                      init_hidden_unit=init_hidden_unit,
                      lam=gae_lambda,
                      clip_param=args.clip_param,
                      entropy_coef=args.entropy_coef, value_coef=args.value_coef)

    checkpoint_path = 'checkpoint/{}_{}_{}_{}_ppo.ckpt'.format(city, temperature_center,
                                                               args.discount, temp_tolerance)

    ppo.train(args.exp_name, env, agent, args.n_iter, args.discount, args.batch_size, np.inf,
              logdir=None, seed=args.seed, checkpoint_path=None)
