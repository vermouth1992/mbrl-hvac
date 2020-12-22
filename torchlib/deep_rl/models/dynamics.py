"""
Model for approximate system dynamics
"""

import torch
import torch.nn as nn

from torchlib.utils.layers import linear_bn_relu_block


class ContinuousMLPDynamics(nn.Module):
    def __init__(self, state_dim, action_dim, nn_size=64):
        super(ContinuousMLPDynamics, self).__init__()
        self.discrete = False
        self.model = nn.Sequential(
            *linear_bn_relu_block(state_dim + action_dim, nn_size, normalize=True),
            *linear_bn_relu_block(nn_size, nn_size, normalize=True),
            nn.Linear(nn_size, state_dim)
        )

    def forward(self, states, actions):
        state_action_input = torch.cat((states, actions), dim=-1)
        return self.model(state_action_input)


class DiscreteMLPDynamics(nn.Module):
    def __init__(self, state_dim, action_dim, nn_size=64):
        super(DiscreteMLPDynamics, self).__init__()
        self.discrete = True
        embedding_dim = action_dim * 5
        self.embedding = nn.Sequential(
            nn.Embedding(action_dim, embedding_dim),
            nn.Dropout(0.1)
        )
        self.model = nn.Sequential(
            *linear_bn_relu_block(state_dim + embedding_dim, nn_size, normalize=True),
            *linear_bn_relu_block(nn_size, nn_size, normalize=True),
            nn.Linear(nn_size, state_dim)
        )

    def forward(self, states, actions):
        actions = self.embedding.forward(actions)
        state_action_input = torch.cat((states, actions), dim=-1)
        return self.model(state_action_input)
