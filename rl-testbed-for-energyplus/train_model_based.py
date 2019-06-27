import numpy as np
from torchlib.deep_rl import RandomAgent
from torchlib.utils.random.sampler import UniformSampler

from agent.agent import VanillaAgent
from agent.model import EnergyPlusDynamicsModel
from agent.planner import BestRandomActionPlanner
from agent.utils import EpisodicHistoryDataset, gather_rollouts
from gym_energyplus.envs.energyplus_env import EnergyPlusEnv
from gym_energyplus.path import get_model_filepath, get_weather_filepath, energyplus_bin_path
from gym_energyplus.wrappers import RepeatAction, EnergyPlusWrapper, Monitor


def train(city='sf',
          temperature_center=22.5,
          temp_tolerance=0.5,
          window_length=20,
          dataset_maxlen=96 * 5 * 4,
          num_init_random_rollouts=2,
          max_rollout_length=96 * 5,  # each
          mpc_horizon=15,
          num_random_action_selection=4096,
          num_on_policy_iters=365 // 5 * 3,
          num_on_policy_rollouts=2,   #
          training_epochs=60,
          training_batch_size=64,
          verbose=True,
          checkpoint_path=None):
    env = EnergyPlusEnv(energyplus_file=energyplus_bin_path,
                        model_file=get_model_filepath('temp_fan'),
                        weather_file=get_weather_filepath(city),
                        config={'temp_center': temperature_center, 'temp_tolerance': temp_tolerance},
                        log_dir=None,
                        verbose=True)
    env = RepeatAction(env)
    env = Monitor(env, log_dir='runs/{}_{}_{}_{}_{}_{}'.format(city, temperature_center, temp_tolerance,
                                                         window_length, mpc_horizon, num_random_action_selection))
    env = EnergyPlusWrapper(env, max_steps=max_rollout_length)

    # collect dataset using random policy
    random_policy = RandomAgent(env.action_space)
    dataset = EpisodicHistoryDataset(maxlen=dataset_maxlen, window_length=window_length)

    print('Gathering initial dataset...')
    initial_dataset = gather_rollouts(env, random_policy, num_init_random_rollouts, np.inf)
    dataset.append(initial_dataset)

    model = EnergyPlusDynamicsModel()

    print('Action space low = {}, high = {}'.format(env.action_space.low, env.action_space.high))

    action_sampler = UniformSampler(low=env.action_space.low, high=env.action_space.high)
    planner = BestRandomActionPlanner(model, action_sampler, env.cost_fn, horizon=mpc_horizon,
                                      num_random_action_selection=num_random_action_selection)

    agent = VanillaAgent(model, planner, window_length, env.action_space)

    agent.set_statistics(dataset)

    agent.train()

    # gather new rollouts using MPC and retrain dynamics model
    for num_iter in range(num_on_policy_iters):
        if verbose:
            print('On policy iteration {}/{}. Size of dataset: {}. Number of trajectories: {}'.format(
                num_iter + 1, num_on_policy_iters, len(dataset), dataset.num_trajectories))
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
