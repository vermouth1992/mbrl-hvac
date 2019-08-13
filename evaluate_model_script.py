import pprint

import numpy as np
import torch
from torchlib.common import map_location
from torchlib.deep_rl import RandomAgent
from torchlib.deep_rl.envs.wrappers import get_wrapper_by_name

from agent import EnergyPlusDynamicsModel
from agent.utils import EpisodicHistoryDataset, gather_rollouts
from gym_energyplus import make_env


def get_model_checkpoint_path(window_length, city):
    return 'checkpoint/deterministic_dynamics_{}_{}.ckpt'.format(window_length, city)


def train_model(window_length, city, num_init_random_rollouts=65):
    env = make_env(city, 23.5, 1.5, obs_normalize=True, num_days_per_episode=1, log_dir=None)

    # collect dataset using random policy
    baseline_agent = RandomAgent(env.action_space)
    dataset = EpisodicHistoryDataset(maxlen=96 * 65, window_length=window_length)

    initial_dataset = gather_rollouts(env, baseline_agent, num_init_random_rollouts, np.inf)
    dataset.append(initial_dataset)

    model = EnergyPlusDynamicsModel(state_dim=env.observation_space.shape[0],
                                    action_dim=env.action_space.shape[0],
                                    hidden_size=32,
                                    learning_rate=1e-3)
    model.set_statistics(dataset)
    model.fit_dynamic_model(dataset, epoch=300, batch_size=128, verbose=True)
    state_dict = {
        'model': model.state_dict(),
        'dataset': dataset
    }
    torch.save(state_dict, get_model_checkpoint_path(window_length, city))


def calculate_h_step_deviation(window_length, city):
    env = make_env(city, 23.5, 1.5, obs_normalize=True, num_days_per_episode=1, log_dir=None)

    model = EnergyPlusDynamicsModel(state_dim=env.observation_space.shape[0],
                                    action_dim=env.action_space.shape[0],
                                    hidden_size=32,
                                    learning_rate=1e-3)

    state_dict = torch.load(get_model_checkpoint_path(window_length, city), map_location=map_location)

    model.load_state_dict(state_dict['model'])
    dataset = state_dict['dataset']

    # predict the states using only first state and actions
    state, action, _ = dataset.random_rollout()

    predicted_states = []

    current_state = state[:window_length]

    for i in range(window_length, state.shape[0]):
        current_action = action[i - window_length:i]
        next_states = model.predict_next_state(current_state, current_action)
        predicted_states.append(next_states)
        current_state = current_state[1:, :]
        current_state = np.concatenate((current_state, np.expand_dims(next_states, axis=0)), axis=0)

    # calculate h step deviation and return the predicted vs. ground truth raw observation
    predicted_states = np.array(predicted_states)
    ground_truth_states = state[window_length:]

    # recover raw state
    gradual_action_wrapper = get_wrapper_by_name(env, 'EnergyPlusGradualActionWrapper')
    obs_normalizer_wrapper = get_wrapper_by_name(env, 'EnergyPlusObsWrapper')

    predicted_states = gradual_action_wrapper.reverse_observation_batch(predicted_states)
    predicted_states = obs_normalizer_wrapper.reverse_observation_batch(predicted_states)

    ground_truth_states = gradual_action_wrapper.reverse_observation_batch(ground_truth_states)
    ground_truth_states = obs_normalizer_wrapper.reverse_observation_batch(ground_truth_states)

    return predicted_states, ground_truth_states


def make_parser():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('action', choices=['train', 'eval'])
    parser.add_argument('--window_length', type=int, default=10)
    parser.add_argument('--city', type=str, default='SF')
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = vars(parser.parse_args())
    pprint.pprint(args)
    if args['action'] == 'train':
        train_model(args['window_length'], city=args['city'])
    else:
        predicted_states, ground_truth_states = calculate_h_step_deviation(args['window_length'], args['city'])
