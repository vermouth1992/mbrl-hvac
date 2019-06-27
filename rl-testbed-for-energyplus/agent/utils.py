from collections import OrderedDict, deque, namedtuple

import numpy as np
from torchlib.dataset.utils import create_data_loader

Transition = namedtuple('Transition', ('state', 'action', 'reward'))
ImitationTransition = namedtuple('ImitationTransition', ('state', 'action', 'reward', 'best_action'))


class EpisodicHistoryDataset(object):
    def __init__(self, maxlen=10000, window_length=20):
        self.memory = deque()
        # current state
        self._states = []
        self._actions = []
        self._rewards = []
        self.size = 0

        self.maxlen = maxlen
        self.window_length = window_length

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
            self.size -= len(trajectory.state)

    def random_iterator(self, batch_size):
        states = []
        actions = []
        delta_states = []
        for trajectory in self.memory:
            for i in range(self.window_length, trajectory.state.shape[0]):
                states.append(trajectory.state[i - self.window_length:i])
                delta_states.append(trajectory.state[i] - trajectory.state[i - 1])
                actions.append(trajectory.action[i - self.window_length:i])

        states = np.stack(states, axis=0)
        actions = np.stack(actions, axis=0)
        delta_states = np.stack(delta_states, axis=0)

        data_loader = create_data_loader((states, actions, delta_states), batch_size=batch_size, shuffle=True,
                                         drop_last=False)

        return data_loader

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


from torchlib.deep_rl import BaseAgent


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
