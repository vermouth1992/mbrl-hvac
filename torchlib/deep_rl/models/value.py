"""
Typical modules for value functions and q values
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchlib.common import FloatTensor
from torchlib.utils.layers import conv2d_bn_relu_block, linear_bn_relu_block, Flatten
from torchlib.utils.weight import fanin_init

"""
Low dimensional classic control module
"""


class QModule(nn.Module):
    def __init__(self, size, state_dim, action_dim):
        super(QModule, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, action_dim)
        )

    def forward(self, state, action=None):
        out = self.model.forward(state)
        if action is not None:
            out = out.gather(1, action.unsqueeze(1)).squeeze()
        return out


class DuelQModule(nn.Module):
    def __init__(self, size, state_dim, action_dim):
        super(DuelQModule, self).__init__()
        self.fc1 = nn.Linear(state_dim, size)
        self.fc2 = nn.Linear(size, size)
        self.adv_fc = nn.Linear(size, action_dim)
        self.value_fc = nn.Linear(size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        value = self.value_fc(x)
        adv = self.adv_fc(x)
        adv = adv - torch.mean(adv, dim=-1, keepdim=True)
        x = value + adv
        return x


class CriticModule(nn.Module):
    def __init__(self, size, state_dim, action_dim):
        super(CriticModule, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, size)
        fanin_init(self.fc1)
        self.fc2 = nn.Linear(size, size)
        fanin_init(self.fc2)
        self.fc3 = nn.Linear(size, 1)
        torch.nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = torch.squeeze(x, dim=-1)
        return x


class DoubleCriticModule(nn.Module):
    def __init__(self, size, state_dim, action_dim):
        super(DoubleCriticModule, self).__init__()
        self.critic1 = CriticModule(size=size, state_dim=state_dim, action_dim=action_dim)
        self.critic2 = CriticModule(size=size, state_dim=state_dim, action_dim=action_dim)

    def forward(self, state, action, minimum=True):
        x1 = self.critic1.forward(state, action)
        x2 = self.critic2.forward(state, action)
        if minimum:
            return torch.min(x1, x2)
        return x1, x2


class DoubleQModule(nn.Module):
    def __init__(self, size, state_dim, action_dim):
        super(DoubleQModule, self).__init__()
        self.critic1 = QModule(size=size, state_dim=state_dim, action_dim=action_dim)
        self.critic2 = QModule(size=size, state_dim=state_dim, action_dim=action_dim)

    def forward(self, state, action=None, minimum=True):
        x1 = self.critic1.forward(state, action)
        x2 = self.critic2.forward(state, action)
        if minimum:
            return torch.min(x1, x2)
        return x1, x2


class ValueModule(nn.Module):
    def __init__(self, size, state_dim):
        super(ValueModule, self).__init__()
        self.fc1 = nn.Linear(state_dim, size)
        fanin_init(self.fc1)
        self.fc2 = nn.Linear(size, size)
        fanin_init(self.fc2)
        self.fc3 = nn.Linear(size, 1)
        torch.nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = torch.squeeze(x, dim=-1)
        return x


"""
Atari Policy
"""


class AtariQModule(nn.Module):
    def __init__(self, frame_history_len, action_dim):
        super(AtariQModule, self).__init__()
        self.model = nn.Sequential(
            *conv2d_bn_relu_block(frame_history_len, 32, kernel_size=8, stride=4, padding=4, normalize=False),
            *conv2d_bn_relu_block(32, 64, kernel_size=4, stride=2, padding=2, normalize=False),
            *conv2d_bn_relu_block(64, 64, kernel_size=3, stride=1, padding=1, normalize=False),
            Flatten(),
            *linear_bn_relu_block(12 * 12 * 64, 512, normalize=False),
            nn.Linear(512, action_dim)
        )

    def forward(self, state, action=None):
        state = state.type(FloatTensor)
        state = state / 255.0
        state = state.permute(0, 3, 1, 2)
        out = self.model.forward(state)
        if action is not None:
            out = out.gather(1, action.unsqueeze(1)).squeeze()
        return out


class DoubleAtariQModule(nn.Module):
    def __init__(self, frame_history_len, action_dim):
        super(DoubleAtariQModule, self).__init__()
        self.critic1 = AtariQModule(frame_history_len, action_dim)
        self.critic2 = AtariQModule(frame_history_len, action_dim)

    def forward(self, state, action=None, minimum=True):
        x1 = self.critic1.forward(state, action)
        x2 = self.critic2.forward(state, action)
        if minimum:
            return torch.min(x1, x2)
        return x1, x2


class AtariDuelQModule(nn.Module):
    def __init__(self, frame_history_len, action_dim):
        super(AtariDuelQModule, self).__init__()
        self.model = nn.Sequential(
            *conv2d_bn_relu_block(frame_history_len, 32, kernel_size=8, stride=4, padding=4, normalize=False),
            *conv2d_bn_relu_block(32, 64, kernel_size=4, stride=2, padding=2, normalize=False),
            *conv2d_bn_relu_block(64, 64, kernel_size=3, stride=1, padding=1, normalize=False),
            Flatten(),
            *linear_bn_relu_block(12 * 12 * 64, 512, normalize=False),
        )
        self.adv_fc = nn.Linear(512, action_dim)
        self.value_fc = nn.Linear(512, 1)

    def forward(self, state, action=None):
        state = state.type(FloatTensor)
        state = state / 255.0
        state = state.permute(0, 3, 1, 2)
        state = self.model.forward(state)
        value = self.value_fc(state)
        adv = self.adv_fc(state)
        adv = adv - torch.mean(adv, dim=-1, keepdim=True)
        out = value + adv
        if action is not None:
            out = out.gather(1, action.unsqueeze(1)).squeeze()
        return out


class DoubleAtariDuelQModule(nn.Module):
    def __init__(self, frame_history_len, action_dim):
        super(DoubleAtariDuelQModule, self).__init__()
        self.critic1 = AtariDuelQModule(frame_history_len, action_dim)
        self.critic2 = AtariDuelQModule(frame_history_len, action_dim)

    def forward(self, state, action=None, minimum=True):
        x1 = self.critic1.forward(state, action)
        x2 = self.critic2.forward(state, action)
        if minimum:
            return torch.min(x1, x2)
        return x1, x2
