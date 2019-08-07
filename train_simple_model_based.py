"""
Collect data using default policies and fit a dynamics model.
Evaluate the dynamics model by using it to plan in real environments without on-policy data aggregation.
1. Raw PID controller.
2. Random exploration under safety constraints

Evaluate the performance by using the learned model to plan on an environment.
"""

import numpy as np
from torchlib.deep_rl import RandomAgent


def gather_dataset(city='sf', agent=None, temperature_center=23.5, temp_tolerance=1.0):
    from gym_energyplus.envs.energyplus_env import EnergyPlusEnv
    from gym_energyplus.path import get_weather_filepath
    from gym_energyplus.wrappers import RepeatAction, EnergyPlusGradualActionWrapper, EnergyPlusObsWrapper, \
        EnergyPlusNormalizeActionWrapper

    config = {
        'temp_center': temperature_center,
        'temp_tolerance': temp_tolerance,
    }

    action_low = np.array([temperature_center - 10., temperature_center - 10., 5., 5.])
    action_high = np.array([temperature_center + 10., temperature_center + 10., 10., 10.])
    action_delta = np.array([2., 2., 5., 5.])

    env = EnergyPlusEnv(weather_file=get_weather_filepath(city),
                        log_dir=None,
                        verbose=True,
                        config=config)
    env = RepeatAction(env)
    env = EnergyPlusNormalizeActionWrapper(env, action_low=action_low, action_high=action_high)
    # env = EnergyPlusObsWrapper(env, temperature_center)

    env.seed(10)

    print(env.action_space.low, env.action_space.high)

    if agent is None:
        agent = RandomAgent(env.action_space)

    state = []
    action = []
    reward = []
    done = False
    obs = env.reset()
    while not done:
        state.append(obs)
        a = agent.predict(obs)
        action.append(a)
        obs, r, done, _ = env.step(a)
        reward.append(r)
    state.append(obs)
    state = np.stack(state, axis=0)
    action = np.stack(action, axis=0)
    np.savez_compressed('data/random_data_{}.npz'.format(city), state=state, action=action, reward=reward)


def train_model(data_path):
    pass


def test_model(model_path):
    pass


if __name__ == '__main__':
    # data = np.load('data/random_data_SF.npz')
    # state = data['state']
    # action = data['action']
    # reward = data['reward']

    gather_dataset('SF', None, 23.5, 1.0)
