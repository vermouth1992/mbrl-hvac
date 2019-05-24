"""
Environment test playground
"""

from gym_energyplus.envs.energyplus_env import EnergyPlusEnv
from gym_energyplus.path import get_model_filepath, get_weather_filepath, energyplus_bin_path

if __name__ == '__main__':
    env = EnergyPlusEnv(energyplus_file=energyplus_bin_path,
                        model_file=get_model_filepath('temp'),
                        weather_file=get_weather_filepath('chicago'),
                        log_dir=None,
                        verbose=True)
