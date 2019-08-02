import numpy as np

from gym_energyplus.envs.energyplus_env import EnergyPlusEnv
from gym_energyplus.path import get_model_filepath, get_weather_filepath, energyplus_bin_path
from gym_energyplus.wrappers import RepeatAction


def test_env_stochasticity():
    """It turns out that it is not a stochastic environment. """
    env = EnergyPlusEnv(energyplus_file=energyplus_bin_path,
                        model_file=get_model_filepath('temp_fan'),
                        weather_file=get_weather_filepath(['SF', 'Tampa']),
                        config={'temp_center': 23.5, 'temp_tolerance': 1.5},
                        log_dir=None,
                        verbose=True)
    env = RepeatAction(env)
    action = np.random.uniform(-1, 1, size=(96 * 365, env.action_space.shape[0]))

    state_0 = []
    reward_0 = []
    state_1 = []
    reward_1 = []

    obs = env.reset()
    state_0.append(obs)
    done = False
    i = 0
    while not done:
        obs, reward, done, info = env.step(action[i])
        i += 1
        state_0.append(obs)
        # print('Step {}'.format(i))
        reward_0.append(reward)

    obs = env.reset()
    state_1.append(obs)
    done = False
    i = 0
    while not done:
        obs, reward, done, info = env.step(action[i])
        i += 1
        state_1.append(obs)
        # print('Step {}'.format(i))
        reward_1.append(reward)

    return np.array(state_0), np.array(reward_0), np.array(state_1), np.array(reward_1)


if __name__ == '__main__':
    # env = EnergyPlusEnv(energyplus_file=energyplus_bin_path,
    #                     model_file=get_model_filepath('temp_fan'),
    #                     weather_file=get_weather_filepath('sf'),
    #                     config={'temp_center': 23.5, 'temp_tolerance': 1.5},
    #                     log_dir=None,
    #                     verbose=True)
    # env = RepeatAction(env)
    # env = EnergyPlusWrapper(env, max_steps=96)
    # env = EnergyPlusDiscreteActionWrapper(env, num_levels=4)
    #
    # _ = env.reset()
    # obs, reward, done, info = env.step(0)
    state_0, reward_0, state_1, reward_1 = test_env_stochasticity()
