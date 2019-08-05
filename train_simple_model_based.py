"""
Collect data using default policies and fit a dynamics model.
Evaluate the dynamics model by using it to plan in real environments without on-policy data aggregation.
1. Raw PID controller.
2. Random exploration under safety constraints

Evaluate the performance by using the learned model to plan on an environment.
"""

import numpy as np
from torchlib.deep_rl import BaseAgent, RandomAgent


class SampleAgent(BaseAgent):
    """
    Collect data in a safe way and ensure the temperature is in safe range.
    """

    def __init__(self, temperature_center, temp_tolerance, safe_action, max_delta_temperature):
        pass


def gather_dataset(city='sf', agent=None, temperature_center=23.5, temp_tolerance=1.0):
    from gym_energyplus.envs.energyplus_env import EnergyPlusEnv
    from gym_energyplus.path import get_weather_filepath
    from gym_energyplus.wrappers import RepeatAction

    config = {
        'temp_center': temperature_center,
        'temp_tolerance': temp_tolerance,
        'safe_action': (-15, 0)
    }
    env = EnergyPlusEnv(weather_file=get_weather_filepath(city),
                        log_dir=None,
                        verbose=True,
                        config=config)
    env = RepeatAction(env)

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


def train_model(data):
    pass


def test_model():
    pass


if __name__ == '__main__':
    gather_dataset('SF', None, 23.5, 1.0)
