"""
Pytorch implementation of proximal policy optimization
"""

import numpy as np
import torch
import torch.nn as nn

from torchlib.common import FloatTensor, move_tensor_to_gpu
from torchlib.dataset.utils import create_data_loader
from .a2c import A2CAgent, get_policy_net
from .utils import compute_reward_to_go_gae


class PPOAgent(A2CAgent):
    def __init__(self, policy_net: nn.Module, policy_optimizer, init_hidden_unit, lam=1.,
                 clip_param=0.2, entropy_coef=0.01, target_kl=0.05,
                 value_coef=1., max_grad_norm=0.5, initial_state_mean=0., initial_state_std=0.):
        super(PPOAgent, self).__init__(policy_net, policy_optimizer, init_hidden_unit, True, lam,
                                       value_coef, max_grad_norm, initial_state_mean, initial_state_std)
        self.target_kl = target_kl
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef

    def compute_old_log_prob(self, observation, hidden, actions):
        with torch.no_grad():
            data_loader = create_data_loader((observation, hidden, actions), batch_size=32, shuffle=False,
                                             drop_last=False)
            old_log_prob = []
            for obs, hid, ac in data_loader:
                obs = move_tensor_to_gpu(obs)
                hid = move_tensor_to_gpu(hid)
                ac = move_tensor_to_gpu(ac)
                old_distribution, _, _ = self.policy_net.forward(obs, hid)
                old_log_prob.append(old_distribution.log_prob(ac))

            old_log_prob = torch.cat(old_log_prob, dim=0).cpu()
        return old_log_prob

    def construct_dataset(self, paths, gamma):

        rewards, advantage, self.state_value_mean, self.state_value_std = compute_reward_to_go_gae(paths, gamma,
                                                                                                   self.policy_net,
                                                                                                   self.lam,
                                                                                                   self.state_value_mean,
                                                                                                   self.state_value_std)

        # reshape all episodes to a single large batch
        observation = np.concatenate([path["observation"] for path in paths])
        hidden = np.concatenate([path["hidden"] for path in paths])
        mask = np.concatenate([path["mask"] for path in paths])
        actions = np.concatenate([path['actions'] for path in paths])

        old_log_prob = self.compute_old_log_prob(observation, hidden, actions)

        return actions, advantage, observation, rewards, old_log_prob, mask

    def update_policy(self, dataset, epoch=4):
        # construct a dataset using paths containing (action, observation, old_log_prob)
        if self.recurrent:
            data_loader = create_data_loader(dataset, batch_size=128, shuffle=False, drop_last=False)
        else:
            data_loader = create_data_loader(dataset, batch_size=128, shuffle=True, drop_last=False)

        for epoch_index in range(epoch):
            current_hidden = torch.tensor(np.expand_dims(self.init_hidden_unit, axis=0),
                                          requires_grad=False).type(FloatTensor)
            for batch_sample in data_loader:
                action, advantage, observation, discount_rewards, old_log_prob, mask = \
                    move_tensor_to_gpu(batch_sample)

                self.policy_optimizer.zero_grad()
                # update policy
                if not self.recurrent:
                    distribution, _, raw_baselines = self.policy_net.forward(observation, None)
                    entropy_loss = distribution.entropy().mean()
                    log_prob = distribution.log_prob(action)
                else:
                    entropy_loss = []
                    log_prob = []
                    raw_baselines = []
                    zero_index = np.where(mask == 0)[0] + 1
                    zero_index = zero_index.tolist()
                    zero_index.insert(0, 0)

                    for i in range(len(zero_index) - 1):
                        start_index = zero_index[i]
                        end_index = zero_index[i + 1]
                        current_obs = observation[start_index:end_index]
                        current_actions = action[start_index:end_index]
                        current_dist, _, current_baseline = self.policy_net.forward(current_obs, current_hidden)
                        current_hidden = torch.tensor(np.expand_dims(self.init_hidden_unit, axis=0),
                                                      requires_grad=False).type(FloatTensor)
                        current_log_prob = current_dist.log_prob(current_actions)

                        log_prob.append(current_log_prob)
                        raw_baselines.append(current_baseline)
                        entropy_loss.append(current_dist.entropy())

                    # last iteration
                    start_index = zero_index[-1]
                    if start_index < observation.shape[0]:
                        current_obs = observation[start_index:]
                        current_actions = action[start_index:]
                        current_dist, current_hidden, current_baseline = self.policy_net.forward(current_obs,
                                                                                                 current_hidden)

                        current_log_prob = current_dist.log_prob(current_actions)

                        log_prob.append(current_log_prob)
                        raw_baselines.append(current_baseline)
                        entropy_loss.append(current_dist.entropy())
                        current_hidden = current_hidden.detach()

                    log_prob = torch.cat(log_prob, dim=0)
                    raw_baselines = torch.cat(raw_baselines, dim=0)
                    entropy_loss = torch.cat(entropy_loss, dim=0).mean()

                assert log_prob.shape == advantage.shape, 'log_prob length {}, advantage length {}'.format(
                    log_prob.shape,
                    advantage.shape)

                # if approximated kl is larger than 1.5 target_kl, we early stop training of this batch
                negative_approx_kl = log_prob - old_log_prob

                negative_approx_kl_mean = torch.mean(-negative_approx_kl).item()

                if negative_approx_kl_mean > 1.5 * self.target_kl:
                    # print('Early stopping this iteration. Current kl {:.4f}. Current epoch index {}'.format(
                    #     negative_approx_kl_mean, epoch_index))
                    continue

                ratio = torch.exp(negative_approx_kl)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = self.get_baseline_loss(raw_baselines, discount_rewards)

                loss = policy_loss - entropy_loss * self.entropy_coef + self.value_coef * value_loss

                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)

                loss.backward()
                self.policy_optimizer.step()


get_policy_net = get_policy_net


def make_default_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.98)
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--target_kl', type=float, default=0.05)
    parser.add_argument('--value_coef', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--recurrent', '-re', action='store_true')
    parser.add_argument('--hidden_size', type=int, default=20)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=2e-3)
    parser.add_argument('--nn_size', '-s', type=int, default=64)
    parser.add_argument('--initial_state_mean', type=float, default=0.)
    parser.add_argument('--initial_state_std', type=float, default=0.)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    return parser
