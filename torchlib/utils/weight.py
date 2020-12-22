"""
A set of utilities for weights. Contains apply a wrapper for a layer and weight init
"""

import numpy as np
import torch.nn as nn


def apply_weight_norm(layer, weight_norm=None):
    if weight_norm:
        layer = weight_norm(layer)
    return layer


def weights_init_normal(m, std=0.02):
    """ This init is common in GAN """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, std)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, std)
        m.bias.data.fill_(0.0)


def kaiming_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def fanin_init(layer):
    fanin = layer.weight.data.size()[0]
    v = 1. / np.sqrt(fanin)
    nn.init.uniform_(layer.weight.data, -v, v)


def xavier_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
