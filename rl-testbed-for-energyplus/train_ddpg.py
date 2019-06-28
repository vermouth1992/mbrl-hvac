"""
Train ddpg for this task
"""

"""
Use DDPG to solve low dimensional control
"""

import os
import pprint

os.environ['CUDA'] = 'False'

import numpy as np
import torch
import torchlib.deep_rl.value_based.ddpg as ddpg
from gym import wrappers
from torchlib.deep_rl.models.value import CriticModule
from torchlib.deep_rl.models.policy import ActorModule
from torchlib.deep_rl.value_based.ddpg import ActorNetwork, CriticNetwork
from torchlib.utils.random.random_process import OrnsteinUhlenbeckActionNoise

from gym_energyplus.envs.energyplus_env import EnergyPlusEnv
from gym_energyplus.path import get_model_filepath, get_weather_filepath, energyplus_bin_path, ENERGYPLUS_WEATHER_dict
from gym_energyplus.wrappers import RepeatAction, EnergyPlusWrapper, Monitor, EnergyPlusObsWrapper


def make_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='ddpg')
    parser.add_argument('--discount', type=float, default=0.95)
    parser.add_argument('--n_iter', '-n', type=int, default=96 * 365 * 29)
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--learning_freq', '-lf', type=int, default=1)
    parser.add_argument('--replay_type', type=str, default='normal')
    parser.add_argument('--replay_size', type=int, default=100000)
    parser.add_argument('--nn_size', '-s', type=int, default=64)
    parser.add_argument('--learn_start', type=int, default=1000)
    parser.add_argument('--target_update_tau', type=float, default=1e-3)
    parser.add_argument('--log_every_n_steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--city', type=str, choices=ENERGYPLUS_WEATHER_dict.keys())
    parser.add_argument('--temp_center', type=float, default=23.5)
    parser.add_argument('--temp_tolerance', type=float, default=1.5)
    return parser


if __name__ == '__main__':
    parser = make_parser()

    args = vars(parser.parse_args())
    pprint.pprint(args)

    city = args['city']
    temperature_center = args['temp_center']
    temp_tolerance = args['temp_tolerance']
    num_days_per_episode = 1

    env = EnergyPlusEnv(energyplus_file=energyplus_bin_path,
                        model_file=get_model_filepath('temp_fan'),
                        weather_file=get_weather_filepath(city),
                        config={'temp_center': temperature_center, 'temp_tolerance': temp_tolerance},
                        log_dir=None,
                        verbose=True)
    env = RepeatAction(env)

    log_dir = 'runs/{}_{}_{}_ddpg'.format(city, temperature_center, temp_tolerance)

    checkpoint_path = 'checkpoint/{}_{}_{}_ddpg.ckpt'.format(city, temperature_center, temp_tolerance)

    env = Monitor(env, log_dir=log_dir)
    env = EnergyPlusWrapper(env, max_steps=96 * num_days_per_episode)
    env = EnergyPlusObsWrapper(env)

    expt_dir = '/tmp/{}'.format('energyplus')
    env = wrappers.Monitor(env, os.path.join(expt_dir, "gym"), force=True, video_callable=False)

    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    action_bound = env.action_space.high
    print('Action space high: {}'.format(env.action_space.high))
    print('Action space low: {}'.format(env.action_space.low))
    assert np.all(env.action_space.high == -env.action_space.low), 'Check the action space.'

    nn_size = args['nn_size']
    tau = args['target_update_tau']

    actor = ActorModule(size=nn_size, state_dim=ob_dim, action_dim=ac_dim, output_activation=torch.tanh)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args['learning_rate'])
    critic = CriticModule(size=nn_size, state_dim=ob_dim, action_dim=ac_dim)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args['learning_rate'])

    actor = ActorNetwork(actor, optimizer=actor_optimizer, tau=tau)
    critic = CriticNetwork(critic, optimizer=critic_optimizer, tau=tau)
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(ac_dim))

    replay_buffer_config = {
        'size': args['replay_size'],
    }

    ddpg.train(env, actor, critic, actor_noise, args['n_iter'], args['replay_type'],
               replay_buffer_config, args['batch_size'], args['discount'], args['learn_start'],
               learning_freq=args['learning_freq'], seed=args['seed'],
               log_every_n_steps=args['log_every_n_steps'],
               actor_checkpoint_path=checkpoint_path)
