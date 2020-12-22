from collections import OrderedDict, deque, namedtuple

import numpy as np
from sklearn.model_selection import train_test_split
from torchlib.dataset.utils import create_data_loader
from torchlib.deep_rl import BaseAgent


class Dataset(object):

    def __init__(self):
        self._states = []
        self._actions = []
        self._next_states = []
        self._rewards = []
        self._dones = []

    @property
    def is_empty(self):
        return len(self) == 0

    def __len__(self):
        return len(self._states)

    ##################
    ### Statistics ###
    ##################

    @property
    def state_mean(self):
        return np.mean(self._states, axis=0).astype(np.float32)

    @property
    def state_std(self):
        return np.std(self._states, axis=0).astype(np.float32)

    @property
    def action_mean(self):
        return np.mean(self._actions, axis=0).astype(np.float32)

    @property
    def action_std(self):
        return np.std(self._actions, axis=0).astype(np.float32)

    @property
    def delta_state_mean(self):
        return np.mean(np.array(self._next_states) - np.array(self._states), axis=0).astype(np.float32)

    @property
    def delta_state_std(self):
        return np.std(np.array(self._next_states) - np.array(self._states), axis=0).astype(np.float32)

    ###################
    ### Adding data ###
    ###################

    def add(self, state, action, next_state, reward, done):
        """
        Add (s, a, r, s') to this dataset
        """
        if not self.is_empty:
            # ensure the state, action, next_state are of the same dimension
            assert len(self._states[-1]) == len(np.ravel(state))
            assert len(self._actions[-1]) == len(np.ravel(action))
            assert len(self._next_states[-1]) == len(np.ravel(next_state))

        self._states.append(np.ravel(state))
        self._actions.append(np.ravel(action))
        self._next_states.append(np.ravel(next_state))
        self._rewards.append(reward)
        self._dones.append(done)

    def append(self, other_dataset):
        """
        Append other_dataset to this dataset
        """
        if not self.is_empty and not other_dataset.is_empty:
            # ensure the state, action, next_state are of the same dimension
            assert len(self._states[-1]) == len(other_dataset._states[-1])
            assert len(self._actions[-1]) == len(other_dataset._actions[-1])
            assert len(self._next_states[-1]) == len(other_dataset._next_states[-1])

        self._states += other_dataset._states
        self._actions += other_dataset._actions
        self._next_states += other_dataset._next_states
        self._rewards += other_dataset._rewards
        self._dones += other_dataset._dones

    ############################
    ### Iterate through data ###
    ############################

    def rollout_iterator(self):
        """
        Iterate through all the rollouts in the dataset sequentially
        """
        end_indices = np.nonzero(self._dones)[0] + 1

        states = np.asarray(self._states)
        actions = np.asarray(self._actions)
        next_states = np.asarray(self._next_states)
        rewards = np.asarray(self._rewards)
        dones = np.asarray(self._dones)

        start_idx = 0
        for end_idx in end_indices:
            indices = np.arange(start_idx, end_idx)
            yield states[indices], actions[indices], next_states[indices], rewards[indices], dones[indices]
            start_idx = end_idx

    def random_iterator(self, batch_size):
        """
        Iterate once through all (s, a, r, s') in batches in a random order
        """
        all_indices = np.nonzero(np.logical_not(self._dones))[0]
        np.random.shuffle(all_indices)

        states = np.asarray(self._states)
        actions = np.asarray(self._actions)
        next_states = np.asarray(self._next_states)
        rewards = np.asarray(self._rewards)
        dones = np.asarray(self._dones)

        i = 0
        while i < len(all_indices):
            indices = all_indices[i:i + batch_size]

            yield states[indices], actions[indices], next_states[indices], rewards[indices], dones[indices]

            i += batch_size

    def log(self):
        end_idxs = np.nonzero(self._dones)[0] + 1

        returns = []

        start_idx = 0
        for end_idx in end_idxs:
            rewards = self._rewards[start_idx:end_idx]
            returns.append(np.sum(rewards))

            start_idx = end_idx

        stats = OrderedDict({
            'ReturnAvg': np.mean(returns),
            'ReturnStd': np.std(returns),
            'ReturnMin': np.min(returns),
            'ReturnMax': np.max(returns)
        })
        return stats


Transition = namedtuple('Transition', ('state', 'action', 'reward'))
ImitationTransition = namedtuple('ImitationTransition', ('state', 'action', 'reward', 'best_action'))


class EpisodicDataset(object):
    def __init__(self, maxlen=10000):
        self.memory = deque()
        # current state
        self._states = []
        self._actions = []
        self._rewards = []
        self.size = 0
        # initial state
        self.initial_state = deque(maxlen=maxlen)

        self.maxlen = maxlen

    def __len__(self):
        return self.size

    @property
    def num_trajectories(self):
        return len(self.memory)

    @property
    def is_empty(self):
        return len(self) == 0

    @property
    def state_mean(self):
        states = []
        for trajectory in self.memory:
            states.append(trajectory.state)
        return np.mean(np.concatenate(states, axis=0), axis=0)

    @property
    def state_std(self):
        states = []
        for trajectory in self.memory:
            states.append(trajectory.state)
        return np.std(np.concatenate(states, axis=0), axis=0)

    @property
    def action_mean(self):
        actions = []
        for trajectory in self.memory:
            actions.append(trajectory.action)
        return np.mean(np.concatenate(actions, axis=0), axis=0).astype(np.float32)

    @property
    def action_std(self):
        actions = []
        for trajectory in self.memory:
            actions.append(trajectory.action)
        return np.std(np.concatenate(actions, axis=0), axis=0).astype(np.float32)

    @property
    def delta_state_mean(self):
        delta_states = []
        for trajectory in self.memory:
            states = trajectory.state
            delta_states.append(states[1:] - states[:-1])
        return np.mean(np.concatenate(delta_states, axis=0), axis=0)

    @property
    def delta_state_std(self):
        delta_states = []
        for trajectory in self.memory:
            states = trajectory.state
            delta_states.append(states[1:] - states[:-1])
        return np.std(np.concatenate(delta_states, axis=0), axis=0)

    @property
    def reward_mean(self):
        rewards = []
        for trajectory in self.memory:
            rewards.append(trajectory.reward)
        return np.mean(np.concatenate(rewards, axis=0), axis=0).astype(np.float32)

    @property
    def reward_std(self):
        rewards = []
        for trajectory in self.memory:
            rewards.append(trajectory.reward)
        return np.std(np.concatenate(rewards, axis=0), axis=0).astype(np.float32)

    def add(self, state, action, next_state, reward, done):
        self._states.append(np.ravel(state))
        if isinstance(action, np.ndarray) and len(action.shape) != 0:
            self._actions.append(np.ravel(action))
        else:
            self._actions.append(action)
        self._rewards.append(np.ravel(reward))

        self.size += 1

        if done:
            self._states.append(next_state)
            self.memory.append(Transition(state=np.array(self._states),
                                          action=np.array(self._actions),
                                          reward=np.array(self._rewards)))
            self._states = []
            self._actions = []
            self._rewards = []

    def append(self, other_dataset):
        self.memory.extend(other_dataset.memory)
        self.size += other_dataset.size

        while self.size > self.maxlen:
            trajectory = self.memory.popleft()
            self.size -= len(trajectory.action)

    def get_initial_states(self):
        init_states = []
        for trajectory in self.memory:
            init_states.append(trajectory.state[0].copy())
        return init_states

    def rollout_iterator(self):
        for trajectory in self.memory:
            states = trajectory.state[:-1]
            next_states = trajectory.state[1:]
            actions = trajectory.action
            rewards = trajectory.reward
            dones = [False] * actions.shape[0]
            dones[-1] = True
            yield states, actions, next_states, rewards, dones

    def random_iterator(self, batch_size, train_val_split_ratio=0.2):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for trajectory in self.memory:
            states.append(trajectory.state[:-1])
            actions.append(trajectory.action)
            next_states.append(trajectory.state[1:])
            rewards.append(trajectory.reward)
            done = [False] * trajectory.action.shape[0]
            done[-1] = True
            dones.append(np.array(done))

        states = np.concatenate(states, axis=0)
        actions = np.concatenate(actions, axis=0)
        next_states = np.concatenate(next_states, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        dones = np.concatenate(dones, axis=0)

        input_tuple = (states, actions, next_states, rewards, dones)

        output_tuple = train_test_split(*input_tuple, test_size=train_val_split_ratio)

        train_tuple = output_tuple[0::2]
        val_tuple = output_tuple[1::2]

        # in training, we drop last batch to avoid batch size 1 that may crash batch_norm layer.
        train_data_loader = create_data_loader(train_tuple, batch_size=batch_size, shuffle=True,
                                               drop_last=True)
        val_data_loader = create_data_loader(val_tuple, batch_size=batch_size, shuffle=False,
                                             drop_last=False)

        return train_data_loader, val_data_loader

    def log(self):
        returns = []

        for trajectory in self.memory:
            returns.append(np.sum(trajectory.reward))

        stats = OrderedDict({
            'ReturnAvg': np.mean(returns),
            'ReturnStd': np.std(returns),
            'ReturnMin': np.min(returns),
            'ReturnMax': np.max(returns)
        })
        return stats


class StateActionPairDataset(object):
    def __init__(self, max_size):
        self.states = deque(maxlen=max_size)
        self.actions = deque(maxlen=max_size)

    def __len__(self):
        return len(self.states)

    @property
    def maxlen(self):
        return self.states.maxlen

    @property
    def is_empty(self):
        return len(self) == 0

    def add(self, state, action):
        self.states.append(state)
        self.actions.append(action)

    @property
    def state_stats(self):
        states = np.array(self.states)
        return np.mean(states, axis=0), np.std(states, axis=0)

    @property
    def action_stats(self):
        actions = np.array(self.actions)
        return np.mean(actions, axis=0), np.std(actions, axis=0)

    def random_iterator(self, batch_size, train_val_split_ratio=0.2):
        states = np.array(self.states)
        actions = np.array(self.actions)

        input_tuple = (states, actions)

        output_tuple = train_test_split(*input_tuple, test_size=train_val_split_ratio)

        train_tuple = output_tuple[0::2]
        val_tuple = output_tuple[1::2]

        # in training, we drop last batch to avoid batch size 1 that may crash batch_norm layer.
        train_data_loader = create_data_loader(train_tuple, batch_size=batch_size, shuffle=True,
                                               drop_last=True)
        val_data_loader = create_data_loader(val_tuple, batch_size=batch_size, shuffle=False,
                                             drop_last=False)

        return train_data_loader, val_data_loader


def gather_rollouts(env, policy: BaseAgent, num_rollouts, max_rollout_length) -> EpisodicDataset:
    dataset = EpisodicDataset()

    for _ in range(num_rollouts):
        state = env.reset()
        done = False
        t = 0
        while not done:
            t += 1

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

    return dataset
