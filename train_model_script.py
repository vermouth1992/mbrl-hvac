import pprint

import numpy as np
import torch
import torch.nn as nn
from torchlib.dataset.utils import create_tuple_data_loader
from torchlib.trainer import Trainer


def gather_dataset(city='sf', mode='temp_fan', num_trajectories=2):
    from gym_energyplus.envs.energyplus_env import EnergyPlusEnv
    from gym_energyplus.path import get_model_filepath, get_weather_filepath, energyplus_bin_path
    from gym_energyplus.wrappers import RepeatAction
    env = EnergyPlusEnv(energyplus_file=energyplus_bin_path,
                        model_file=get_model_filepath(mode),
                        weather_file=get_weather_filepath(city),
                        log_dir=None,
                        verbose=True)
    env = RepeatAction(env)
    for i in range(num_trajectories):
        state = []
        action = []
        reward = 0
        done = False
        obs = env.reset()
        while not done:
            state.append(obs)
            a = (np.random.rand(env.action_space.low.shape[0]) - 0.5) * 2
            action.append(a)
            obs, r, done, _ = env.step(a)
            reward += r
        state.append(obs)
        state = np.stack(state, axis=0)
        action = np.stack(action, axis=0)
        np.savez_compressed('data/random_data_{}_{}_{}.npz'.format(mode, city, i), state=state, action=action)


def create_dataset(state, action, window_length=10):
    """
    state: (num_times + 1, ob_dim),
    action: (num_times, ob_dim)
    """
    state_x = []
    action_x = []
    delta_state_y = []
    for i in range(action.shape[0] - window_length):
        state_x.append(state[i:i + window_length, :])
        action_x.append(action[i:i + window_length, :])
        delta_state_y.append(state[i + window_length, :] - state[i + window_length - 1, :])
    state_x = np.stack(state_x, axis=0)
    action_x = np.stack(action_x, axis=0)
    delta_state_y = np.stack(delta_state_y, axis=0)
    return state_x, action_x, delta_state_y


def train(window_length, city):
    data_0 = np.load('data/random_data_temp_fan_{}_0.npz'.format(city))
    state = data_0['state']
    action = data_0['action']

    # create data statistics
    state_mean = np.mean(state, axis=0)
    state_std = np.std(state, axis=0)
    action_mean = np.mean(action, axis=0)
    action_std = np.std(action, axis=0)

    state_x, action_x, delta_state_y = create_dataset(state, action, window_length=window_length)

    delta_state_mean = np.mean(delta_state_y, axis=0)
    delta_state_std = np.std(delta_state_y, axis=0)

    # normalize each data with mean and std
    state_x = (state_x - state_mean) / state_std
    action_x = (action_x - action_mean) / action_std
    delta_state_y = (delta_state_y - delta_state_mean) / delta_state_std

    # change data type to float32
    state_x = state_x.astype(np.float32)
    action_x = action_x.astype(np.float32)
    delta_state_y = delta_state_y.astype(np.float32)

    # split training data and validation data
    from sklearn.model_selection import train_test_split

    state_x_train, state_x_val, action_x_train, action_x_val, delta_state_y_train, delta_state_y_val = train_test_split(
        state_x, action_x, delta_state_y, test_size=0.25)

    # create data loader
    train_data_loader = create_tuple_data_loader(((state_x_train, action_x_train), (delta_state_y_train,)))
    val_data_loader = create_tuple_data_loader(((state_x_val, action_x_val), (delta_state_y_val,)))

    # define model
    from agent.model import LSTMAttention

    model = LSTMAttention(state_dim=state.shape[-1], action_dim=action.shape[-1], hidden_size=32)

    # define optimizer and loss

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)
    loss = [nn.MSELoss()]

    # define trainer
    trainer = Trainer(model, optimizer, loss, metrics=None, scheduler=scheduler)

    checkpoint_path = 'checkpoint/lstm_attention_{}_{}.th'.format(window_length, city)

    trainer.fit(train_data_loader=train_data_loader, epochs=150, val_data_loader=val_data_loader,
                model_path=checkpoint_path)

    np.savez_compressed('checkpoint/lstm_attention_{}_{}_stats.npz'.format(window_length, city),
                        state_mean=state_mean,
                        state_std=state_std,
                        action_mean=action_mean,
                        action_std=action_std,
                        delta_state_mean=delta_state_mean,
                        delta_state_std=delta_state_std)


def make_parser():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('action', choices=['gather', 'train'])
    parser.add_argument('--window_length', type=int, default=10)
    parser.add_argument('--city', type=str, default='sf')
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = vars(parser.parse_args())
    pprint.pprint(args)

    if args['action'] == 'gather':
        gather_dataset(city=args['city'])
    else:
        train(args['window_length'], args['city'])
