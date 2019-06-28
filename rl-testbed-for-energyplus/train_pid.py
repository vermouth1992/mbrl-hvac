"""
Environment test playground
"""

import numpy as np
from torchlib.deep_rl import BaseAgent

from gym_energyplus.envs.energyplus_env import EnergyPlusEnv
from gym_energyplus.path import get_model_filepath, get_weather_filepath, energyplus_bin_path
from gym_energyplus.wrappers import RepeatAction, EnergyPlusWrapper, Monitor


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
        return self.normalize_action(action)


outside_temp_difference = {
    'chicago': 3.5,
    'sf': 3.5,
    'sterling': 3.5,
    'golden': 3.5
}


def make_parser():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--city', type=str, choices=outside_temp_difference.keys())
    parser.add_argument('--temp_center', type=float, default=23.5)
    parser.add_argument('--temp_tolerance', type=float, default=1.5)
    return parser


if __name__ == '__main__':

    parser = make_parser()
    args = vars(parser.parse_args())

    import pprint

    pprint.pprint(args)

    city = args['city']

    temperature_center = args['temp_center']
    temperature_tolerance = args['temp_tolerance']

    env = EnergyPlusEnv(energyplus_file=energyplus_bin_path,
                        model_file=get_model_filepath('temp_fan'),
                        weather_file=get_weather_filepath(city),
                        config={'temp_center': temperature_center, 'temp_tolerance': temperature_tolerance},
                        log_dir=None,
                        verbose=True)
    env = RepeatAction(env)
    env = Monitor(env, log_dir='runs/{}_{}_{}_pid'.format(city, temperature_center, temperature_tolerance))
    env = EnergyPlusWrapper(env, max_steps=96)

    true_done = False
    day_index = 1

    agent = PIDAgent(target=temperature_center - outside_temp_difference[city], sensitivity=1.0, alpha=0.5)

    while not true_done:
        obs = env.reset()
        print('Day {}'.format(day_index))
        done = False
        while not done:
            action = agent.predict(obs)
            obs, reward, done, info = env.step(action)
        day_index += 1
        true_done = info['true_done']
