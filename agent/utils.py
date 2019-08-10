import numpy as np
from sklearn.model_selection import train_test_split
from torchlib.dataset.utils import create_data_loader
from torchlib.deep_rl import BaseAgent
from torchlib.deep_rl.algorithm.model_based.utils import EpisodicDataset


class EpisodicHistoryDataset(EpisodicDataset):
    def __init__(self, maxlen=10000, window_length=20):
        super(EpisodicHistoryDataset, self).__init__(maxlen=maxlen)
        self.window_length = window_length

    @property
    def state_mean(self):
        states = []
        for trajectory in self.memory:
            states.append(trajectory.state)
        return np.mean(np.concatenate(states, axis=0), axis=0, keepdims=True)

    @property
    def state_std(self):
        states = []
        for trajectory in self.memory:
            states.append(trajectory.state)
        return np.std(np.concatenate(states, axis=0), axis=0, keepdims=True)

    @property
    def action_mean(self):
        actions = []
        for trajectory in self.memory:
            actions.append(trajectory.action)
        return np.mean(np.concatenate(actions, axis=0), axis=0, keepdims=True)

    @property
    def action_std(self):
        actions = []
        for trajectory in self.memory:
            actions.append(trajectory.action)
        return np.std(np.concatenate(actions, axis=0), axis=0, keepdims=True)

    def random_iterator(self, batch_size, train_val_split_ratio=0.2):
        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []
        for trajectory in self.memory:
            for i in range(self.window_length, trajectory.state.shape[0]):
                states.append(trajectory.state[i - self.window_length:i])
                next_states.append(trajectory.state[i])
                actions.append(trajectory.action[i - self.window_length:i])
            rewards.append(trajectory.reward[self.window_length - 1:])
            done = [False] * (trajectory.action.shape[0] - self.window_length + 1)
            done[-1] = True
            dones.append(np.array(done))

        states = np.stack(states, axis=0)
        actions = np.stack(actions, axis=0)
        next_states = np.stack(next_states, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        dones = np.concatenate(dones, axis=0)

        input_tuple = (states, actions, next_states, rewards, dones)

        output_tuple = train_test_split(*input_tuple, test_size=train_val_split_ratio)

        train_tuple = output_tuple[0::2]
        val_tuple = output_tuple[1::2]

        train_data_loader = create_data_loader(train_tuple, batch_size=batch_size, shuffle=True,
                                               drop_last=False)
        val_data_loader = create_data_loader(val_tuple, batch_size=batch_size, shuffle=True,
                                             drop_last=False)

        return train_data_loader, val_data_loader


def gather_rollouts(env, policy: BaseAgent, num_rollouts, max_rollout_length):
    dataset = EpisodicHistoryDataset()

    for _ in range(num_rollouts):
        state = env.reset()
        done = False
        t = 0
        while not done:
            if state.dtype == np.float:
                state = state.astype(np.float32)

            action = policy.predict(state)

            if isinstance(action, np.ndarray) and action.dtype == np.float:
                action = action.astype(np.float32)

            next_state, reward, done, _ = env.step(action)

            if next_state.dtype == np.float:
                next_state = next_state.astype(np.float32)

            done = done or (t >= max_rollout_length)

            dataset.add(state, action, next_state, reward, done)

            state = next_state
            t += 1

    return dataset
