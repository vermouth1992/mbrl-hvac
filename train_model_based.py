import numpy as np
from torchlib.deep_rl import RandomAgent
from torchlib.utils.random.sampler import UniformSampler

from agent import ModelBasedHistoryPlanAgent, ModelBasedHistoryDaggerAgent, EnergyPlusDynamicsModel, \
    BestRandomActionHistoryPlanner
from agent.utils import EpisodicHistoryDataset, gather_rollouts
from gym_energyplus import make_env, ALL_CITIES


def train(city=('SF'),
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
          dagger=False,
          verbose=True,
          checkpoint_path=None):
    dataset_maxlen = 96 * num_days_per_episodes * num_dataset_maxlen_days  # the dataset contains 8 weeks of historical data
    max_rollout_length = 96 * num_days_per_episodes  # each episode is n days
    num_on_policy_iters = (365 * num_years // num_days_per_episodes -
                           num_init_random_rollouts) // num_on_policy_rollouts

    log_dir = 'runs/{}_{}_{}_{}_{}_{}_{}_{}_model_based'.format('_'.join(city), temperature_center, temp_tolerance,
                                                                window_length, mpc_horizon,
                                                                num_random_action_selection,
                                                                num_on_policy_rollouts,
                                                                training_epochs)

    env = make_env(city, temperature_center, temp_tolerance, obs_normalize=True,
                   num_days_per_episode=1, log_dir=log_dir)

    # collect dataset using random policy
    baseline_agent = RandomAgent(env.action_space)
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
    planner = BestRandomActionHistoryPlanner(model, action_sampler, env.cost_fn, horizon=mpc_horizon,
                                             num_random_action_selection=num_random_action_selection,
                                             gamma=gamma)
    if dagger:
        agent = ModelBasedHistoryDaggerAgent(model=model, planner=planner, policy_data_size=1000,
                                             window_length=window_length, baseline_agent=baseline_agent,
                                             state_dim=env.observation_space.shape[0],
                                             action_dim=env.action_space.shape[0],
                                             hidden_size=32)
    else:
        agent = ModelBasedHistoryPlanAgent(model, planner, window_length, baseline_agent)

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
    parser.add_argument('--city', type=str, choices=ALL_CITIES, nargs='+')
    parser.add_argument('--temp_center', type=float, default=23.5)
    parser.add_argument('--temp_tolerance', type=float, default=1.5)
    parser.add_argument('--window_length', type=int, default=20)
    parser.add_argument('--num_years', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--num_init_random_rollouts', type=int, default=65)
    parser.add_argument('--training_batch_size', type=int, default=128)
    parser.add_argument('--num_dataset_maxlen_days', type=int, default=120)
    parser.add_argument('--num_random_action_selection', type=int, default=8192)

    parser.add_argument('--mpc_horizon', type=int, default=5)
    parser.add_argument('--num_days_on_policy', type=int, default=15)
    parser.add_argument('--training_epochs', type=int, default=60)
    parser.add_argument('--dagger', action='store_true')

    return parser


if __name__ == '__main__':
    parser = make_parser()

    args = vars(parser.parse_args())

    import pprint

    pprint.pprint(args)

    train(city=args['city'],
          temperature_center=args['temp_center'],
          temp_tolerance=args['temp_tolerance'],
          window_length=args['window_length'],
          num_dataset_maxlen_days=args['num_dataset_maxlen_days'],
          num_days_per_episodes=1,
          num_init_random_rollouts=args['num_init_random_rollouts'],  # 56 days as initial period
          num_on_policy_rollouts=args['num_days_on_policy'],
          # 5 days as grace period, indicated as data distribution shift
          num_years=args['num_years'],
          mpc_horizon=args['mpc_horizon'],
          gamma=args['gamma'],
          num_random_action_selection=args['num_random_action_selection'],
          training_epochs=args['training_epochs'],
          training_batch_size=args['training_batch_size'],
          dagger=args['dagger'],
          verbose=True,
          checkpoint_path=None)
