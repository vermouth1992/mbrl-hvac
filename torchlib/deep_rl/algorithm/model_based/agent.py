"""
Implement model-based reinforcement learning in https://arxiv.org/abs/1708.02596
Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning
The steps are:
1. Collect random dataset (s, a, s', r) using random policy.
2. Train an initial dynamic model.
2. Fine-tune by using on policy data.
"""

import torch
from gym import Env

from torchlib.common import map_location
from torchlib.deep_rl import BaseAgent, RandomAgent
from .planner import Planner
from .policy import ImitationPolicy
from .utils import EpisodicDataset as Dataset, StateActionPairDataset, gather_rollouts
from .world_model import WorldModel


class ModelBasedAgent(BaseAgent):
    """
    In vanilla agent, it trains a world model and using the world model to plan.
    """

    def __init__(self, model: WorldModel):
        self.model = model

    def save_checkpoint(self, checkpoint_path):
        print('Saving checkpoint to {}'.format(checkpoint_path))
        torch.save(self.model.state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        states = torch.load(checkpoint_path, map_location=map_location)
        self.model.load_state_dict(states)

    def set_statistics(self, initial_dataset: Dataset):
        self.model.set_statistics(initial_dataset)

    def predict(self, state):
        raise NotImplementedError

    def fit_dynamic_model(self, dataset: Dataset, epoch=10, batch_size=128, verbose=False):
        self.model.train()
        self.model.fit_dynamic_model(dataset, epoch, batch_size, verbose)

    def fit_policy(self, dataset: Dataset, epoch=10, batch_size=128, verbose=False):
        raise NotImplementedError

    def train(self, env: Env,
              dataset_maxlen=10000,
              num_init_random_rollouts=10,
              max_rollout_length=500,
              num_on_policy_iters=10,
              num_on_policy_rollouts=10,
              model_training_epochs=60,
              policy_training_epochs=60,
              training_batch_size=512,
              default_policy=None,
              verbose=True,
              checkpoint_path=None):
        # collect dataset using random policy
        if default_policy is None:
            default_policy = RandomAgent(env.action_space)
        dataset = Dataset(maxlen=dataset_maxlen)

        print('Gathering initial dataset...')

        if max_rollout_length <= 0:
            max_rollout_length = env.spec.max_episode_steps

        initial_dataset = gather_rollouts(env, default_policy, num_init_random_rollouts, max_rollout_length)
        dataset.append(initial_dataset)

        # gather new rollouts using MPC and retrain dynamics model
        for num_iter in range(num_on_policy_iters):
            if verbose:
                print('On policy iteration {}/{}. Size of dataset: {}. Number of trajectories: {}'.format(
                    num_iter + 1, num_on_policy_iters, len(dataset), dataset.num_trajectories))

            self.set_statistics(dataset)
            self.fit_dynamic_model(dataset=dataset, epoch=model_training_epochs, batch_size=training_batch_size,
                                   verbose=verbose)
            self.fit_policy(dataset=dataset, epoch=policy_training_epochs, batch_size=training_batch_size,
                            verbose=verbose)
            on_policy_dataset = gather_rollouts(env, self, num_on_policy_rollouts, max_rollout_length)

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
            self.save_checkpoint(checkpoint_path)


class ModelBasedPlanAgent(ModelBasedAgent):
    def __init__(self, model: WorldModel, planner: Planner):
        super(ModelBasedPlanAgent, self).__init__(model=model)
        self.planner = planner

    def predict(self, state):
        self.model.eval()
        return self.planner.predict(state)

    def fit_policy(self, dataset: Dataset, epoch=10, batch_size=128, verbose=False):
        pass


class ModelBasedDAggerAgent(ModelBasedPlanAgent):
    """
    Imitate optimal action by training a policy model using DAgger
    """

    def __init__(self, model, planner, policy: ImitationPolicy, policy_data_size=1000):
        super(ModelBasedDAggerAgent, self).__init__(model=model, planner=planner)
        self.policy = policy

        self.state_action_dataset = StateActionPairDataset(max_size=policy_data_size)

    def save_checkpoint(self, checkpoint_path):
        print('Saving checkpoint to {}'.format(checkpoint_path))
        states = {
            'model': self.model.state_dict,
            'policy': self.policy.state_dict
        }
        torch.save(states, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        states = torch.load(checkpoint_path, map_location=map_location)
        self.model.load_state_dict(states['model'])
        self.policy.load_state_dict(states['policy'])

    def set_statistics(self, initial_dataset: Dataset):
        """ Set statistics for model and policy

        Args:
            initial_dataset: dataset collected by initial (random) policy

        Returns: None

        """
        super(ModelBasedDAggerAgent, self).set_statistics(initial_dataset=initial_dataset)
        self.policy.set_state_stats(initial_dataset.state_mean, initial_dataset.state_std)

    def predict(self, state):
        """ When collecting on policy data, we also bookkeeping optimal state, action pair
            (s, a) for training dagger model.

        Args:
            state: (state_dim,)

        Returns: (ac_dim,)

        """
        action = super(ModelBasedDAggerAgent, self).predict(state=state)
        self.state_action_dataset.add(state=state, action=action)
        self.policy.eval()
        action = self.policy.predict(state)
        return action

    def fit_policy(self, dataset: Dataset, epoch=10, batch_size=128, verbose=False):
        if len(self.state_action_dataset) > 0:
            self.policy.train()
            self.policy.fit(self.state_action_dataset, epoch=epoch, batch_size=batch_size,
                            verbose=verbose)
