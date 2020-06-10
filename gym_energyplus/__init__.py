# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Reinforcement Learning Testbed for Power Consumption Optimization
# This project is licensed under the MIT License, see LICENSE

from gym.envs.registration import register

register(
    id='EnergyPlus-v0',
    entry_point='gym_energyplus.envs:EnergyPlusEnv',
)

import numpy as np
from .envs.energyplus_env import EnergyPlusEnv
from .path import get_model_filepath, get_weather_filepath, energyplus_bin_path, ENERGYPLUS_WEATHER_dict
from .wrappers import RepeatAction, EnergyPlusSplitEpisodeWrapper, Monitor, EnergyPlusObsWrapper, \
    EnergyPlusNormalizeActionWrapper, EnergyPlusGradualActionWrapper

ALL_CITIES = set(ENERGYPLUS_WEATHER_dict.keys())


def make_env(cities, temperature_center, temp_tolerance, obs_normalize=True, action_normalize=True,
             num_days_per_episode=1, window_length=None, log_dir=None):
    env = EnergyPlusEnv(energyplus_file=energyplus_bin_path,
                        model_file=get_model_filepath('temp_fan'),
                        weather_file=get_weather_filepath(cities),
                        config={'temp_center': temperature_center, 'temp_tolerance': temp_tolerance},
                        log_dir=None,
                        verbose=True)
    # env = RepeatAction(env)

    assert 365 % num_days_per_episode == 0, '365 must be divisible by num_days_per_episode. Got {}'.format(
        num_days_per_episode)

    if log_dir is not None:
        env = Monitor(env, log_dir=log_dir)

    if obs_normalize:
        env = EnergyPlusObsWrapper(env, temperature_center)

    if action_normalize:
        action_low = np.array([temperature_center - 15., temperature_center - 15., 2.5, 2.5])
        action_high = np.array([temperature_center + 0., temperature_center + 0., 10., 10.])
        action_delta = np.array([1.0, 1.0, 1.0, 1.0])

        # env = EnergyPlusNormalizeActionWrapper(env=env, action_low=action_low, action_high=action_high)
        env = EnergyPlusGradualActionWrapper(env=env, action_low=action_low, action_high=action_high,
                                             action_delta=action_delta)

    env = EnergyPlusSplitEpisodeWrapper(env, max_steps=96 * num_days_per_episode, window_length=window_length)

    return env
