import os
import shutil

import numpy as np
from torchlib.deep_rl import BaseAgent
from torchlib.utils.random.sampler import UniformSampler

from agent.agent import VanillaAgent
from agent.model import EnergyPlusDynamicsModel
from agent.planner import BestRandomActionPlanner
from agent.utils import EpisodicHistoryDataset, gather_rollouts
from gym_energyplus.envs.energyplus_env import EnergyPlusEnv
from gym_energyplus.path import get_model_filepath, get_weather_filepath, energyplus_bin_path, ENERGYPLUS_WEATHER_dict
from gym_energyplus.wrappers import RepeatAction, EnergyPlusWrapper, Monitor, EnergyPlusObsWrapper


class PIDAgent(BaseAgent):
    def __init__(self, target, sensitivity=1.0, alpha=0.4):
        self.sensitivity = sensitivity
        self.act_west_prev = target
        self.act_east_prev = target
        self.alpha = alpha
        self.target = target

        self.lo = 10.0
        self.hi = 40.0
        self.flow_hi = 7.0
        self.flow_lo = self.flow_hi * 0.25

        self.default_flow = self.flow_hi

        self.low = np.array([self.lo, self.lo, self.flow_lo, self.flow_lo])
        self.high = np.array([self.hi, self.hi, self.flow_hi, self.flow_hi])

    def normalize_action(self, action):
        action = ((action - self.low) / (self.high - self.low) - 0.5) * 2.
        return action

    def predict(self, state):
        delta_west = state[1] - self.target
        act_west = self.target - delta_west * self.sensitivity
        act_west = act_west * self.alpha + self.act_west_prev * (1 - self.alpha)
        self.act_west_prev = act_west

        delta_east = state[2] - self.target
        act_east = self.target - delta_east * self.sensitivity
        act_east = act_east * self.alpha + self.act_east_prev * (1 - self.alpha)
        self.act_east_prev = act_east

        act_west = max(self.lo, min(act_west, self.hi))
        act_east = max(self.lo, min(act_east, self.hi))
        action = np.array([act_west, act_east, self.default_flow, self.default_flow])
        return self.normalize_action(action).astype(np.float32)


def train(city='sf',
          temperature_center=22.5,
          temp_tolerance=0.5,
          window_length=20,
          num_dataset_maxlen_days=56,
          num_days_per_episodes=5,
          num_init_random_rollouts=2,  # 10 days as initial period
          num_on_policy_rollouts=2,  # 10 days as grace period, indicated as data distribution shift
          num_years=3,
          mpc_horizon=15,
          gamma=0.95,
          num_random_action_selection=4096,
          training_epochs=60,
          training_batch_size=128,
          verbose=True,
          checkpoint_path=None):
    dataset_maxlen = 96 * num_days_per_episodes * num_dataset_maxlen_days  # the dataset contains 8 weeks of historical data
    max_rollout_length = 96 * num_days_per_episodes  # each episode is n days
    num_on_policy_iters = 365 // num_days_per_episodes // num_on_policy_rollouts * num_years

    env = EnergyPlusEnv(energyplus_file=energyplus_bin_path,
                        model_file=get_model_filepath('temp_fan'),
                        weather_file=get_weather_filepath(city),
                        config={'temp_center': temperature_center, 'temp_tolerance': temp_tolerance},
                        log_dir=None,
                        verbose=True)
    env = RepeatAction(env)

    log_dir = 'runs/{}_{}_{}_{}_{}_{}_{}_{}_model_based'.format(city, temperature_center, temp_tolerance,
                                                                window_length, mpc_horizon,
                                                                num_random_action_selection,
                                                                num_on_policy_rollouts,
                                                                training_epochs)
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    env = Monitor(env, log_dir=log_dir)
    env = EnergyPlusWrapper(env, max_steps=max_rollout_length)
    env = EnergyPlusObsWrapper(env)

    # collect dataset using random policy
    baseline_agent = PIDAgent(target=temperature_center - 3.5)
    dataset = EpisodicHistoryDataset(maxlen=dataset_maxlen, window_length=window_length)

    print('Gathering initial dataset...')
    initial_dataset = gather_rollouts(env, baseline_agent, num_init_random_rollouts, np.inf)
    dataset.append(initial_dataset)

    model = EnergyPlusDynamicsModel(state_dim=env.observation_space.shape[0],
                                    action_dim=env.action_space.shape[0],
                                    hidden_size=32,
                                    learning_rate=1e-3)

    print('Action space low = {}, high = {}'.format(env.action_space.low, env.action_space.high))

    action_sampler = UniformSampler(low=env.action_space.low, high=env.action_space.high)
    planner = BestRandomActionPlanner(model, action_sampler, env.cost_fn, horizon=mpc_horizon,
                                      num_random_action_selection=num_random_action_selection,
                                      gamma=gamma)

    agent = VanillaAgent(model, planner, window_length, baseline_agent)

    # gather new rollouts using MPC and retrain dynamics model
    for num_iter in range(num_on_policy_iters):
        if verbose:
            print('On policy iteration {}/{}. Size of dataset: {}. Number of trajectories: {}'.format(
                num_iter + 1, num_on_policy_iters, len(dataset), dataset.num_trajectories))
        agent.set_statistics(dataset)
        agent.fit_dynamic_model(dataset=dataset, epoch=training_epochs, batch_size=training_batch_size,
                                verbose=verbose)
        on_policy_dataset = gather_rollouts(env, agent, num_on_policy_rollouts, max_rollout_length)

        # record on policy dataset statistics
        if verbose:
            stats = on_policy_dataset.log()
            strings = []
            for key, value in stats.items():
                strings.append(key + ": {:.4f}".format(value))
            strings = " - ".join(strings)
            print(strings)

        dataset.append(on_policy_dataset)

    if checkpoint_path:
        agent.save_checkpoint(checkpoint_path)


def make_parser():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--city', type=str, choices=ENERGYPLUS_WEATHER_dict.keys())
    parser.add_argument('--temp_center', type=float, default=23.5)
    parser.add_argument('--temp_tolerance', type=float, default=1.5)
    parser.add_argument('--window_length', type=int, default=20)
    parser.add_argument('--num_years', type=int, default=5)
    parser.add_argument('--num_days_on_policy', type=int, default=10)
    parser.add_argument('--mpc_horizon', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--training_epochs', type=int, default=60)
    parser.add_argument('--training_batch_size', type=int, default=128)
    return parser

if __name__ == '__main__':
    parser = make_parser()

    args = vars(parser.parse_args())

    import pprint

    pprint.pprint(args)

    city = args['city']
    temperature_center = args['temp_center']
    temp_tolerance = args['temp_tolerance']
    window_length = args['window_length']

    train(city=city,
          temperature_center=temperature_center,
          temp_tolerance=temp_tolerance,
          window_length=window_length,
          num_dataset_maxlen_days=120,
          num_days_per_episodes=1,
          num_init_random_rollouts=60,  # 56 days as initial period
          num_on_policy_rollouts=args['num_days_on_policy'],
          # 5 days as grace period, indicated as data distribution shift
          num_years=args['num_years'],
          mpc_horizon=args['mpc_horizon'],
          gamma=args['gamma'],
          num_random_action_selection=8192,
          training_epochs=args['training_epochs'],
          training_batch_size=args['training_batch_size'],
          verbose=True,
          checkpoint_path=None)
