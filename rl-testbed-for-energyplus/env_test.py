"""
Environment test playground
"""

from gym_energyplus.envs.energyplus_env import EnergyPlusEnv
from gym_energyplus.path import get_model_filepath, get_weather_filepath, energyplus_bin_path
from gym_energyplus.wrappers import RepeatAction, EnergyPlusWrapper

if __name__ == '__main__':
    env = EnergyPlusEnv(energyplus_file=energyplus_bin_path,
                        model_file=get_model_filepath('temp_fan'),
                        weather_file=get_weather_filepath('chicago'),
                        config={'temp_center': 22.5, 'temp_tolerance': 1.0},
                        log_dir=None,
                        verbose=True)
    env = RepeatAction(env)
    env = EnergyPlusWrapper(env)
