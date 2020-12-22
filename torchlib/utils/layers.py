"""
A set of pytorch layer utilities for fast network building
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchlib.common import enable_cuda
from torchlib.utils.weight import apply_weight_norm

"""
Basic layer utils
"""


def linear_relu_block(in_feat, out_feat):
    layers = [nn.Linear(in_feat, out_feat)]
    layers.append(nn.ReLU(inplace=True))
    return layers


def linear_lrelu_block(in_feat, out_feat, alpha=0.2):
    layers = [nn.Linear(in_feat, out_feat)]
    layers.append(nn.LeakyReLU(alpha, inplace=True))
    return layers


def linear_bn_relu_block(in_feat, out_feat, normalize=True):
    """ linear + batchnorm + leaky relu """
    layers = [nn.Linear(in_feat, out_feat)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_feat))
    layers.append(nn.ReLU(inplace=True))
    return layers


def linear_bn_relu_dropout_block(in_feat, out_feat, normalize=True, p=0.5):
    layers = [nn.Linear(in_feat, out_feat)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_feat))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Dropout(p))
    return layers


def linear_bn_lrelu_block(in_feat, out_feat, normalize=True, alpha=0.2):
    """ linear + batchnorm + leaky relu """
    layers = [nn.Linear(in_feat, out_feat)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_feat))
    layers.append(nn.LeakyReLU(alpha, inplace=True))
    return layers


def linear_bn_lrelu_dropout_block(in_feat, out_feat, normalize=True, alpha=0.2, p=0.5):
    """ linear + batchnorm + leaky relu """
    layers = [nn.Linear(in_feat, out_feat)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_feat))
    layers.append(nn.LeakyReLU(alpha, inplace=True))
    layers.append(nn.Dropout(p))
    return layers


def conv1d_bn_relu_block(in_channels, out_channels, kernel_size, stride, padding, normalize=True, bias=True,
                         weight_norm=None):
    conv = apply_weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias), weight_norm)
    layers = [conv]
    if normalize:
        layers.append(nn.BatchNorm1d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    return layers


def conv2d_bn_relu_block(in_channels, out_channels, kernel_size, stride, padding, normalize=True, bias=True,
                         weight_norm=None):
    """ conv2d + batchnorm (optional) + relu """
    conv = apply_weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias), weight_norm)
    layers = [conv]
    if normalize:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    return layers


def conv1d_bn_lrelu_block(in_channels, out_channels, kernel_size, stride, padding, alpha=0.2, normalize=True, bias=True,
                          weight_norm=None):
    conv = apply_weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias), weight_norm)
    layers = [conv]
    if normalize:
        layers.append(nn.BatchNorm1d(out_channels))
    layers.append(nn.LeakyReLU(alpha, inplace=True))
    return layers


def conv2d_bn_lrelu_block(in_channels, out_channels, kernel_size, stride, padding, alpha=0.2, normalize=True,
                          bias=True, weight_norm=None):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
    if weight_norm:
        conv = weight_norm(conv)
    layers = [conv]
    if normalize:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(alpha, inplace=True))
    return layers


def conv2d_bn_lrelu_dropout_block(in_channels, out_channels, kernel_size, stride, padding, alpha=0.2, p=0.5,
                                  normalize=True, bias=True, weight_norm=None):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
    if weight_norm:
        conv = weight_norm(conv)
    layers = [conv]
    if normalize:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(alpha, inplace=True))
    layers.append(nn.Dropout(p))
    return layers


def conv1d_trans_bn_relu_block(in_channels, out_channels, kernel_size, stride, padding, normalize=True, bias=True,
                               weight_norm=None):
    conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
    if weight_norm:
        conv = weight_norm(conv)
    layers = [conv]
    if normalize:
        layers.append(nn.BatchNorm1d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    return layers


def conv2d_trans_bn_relu_block(in_channels, out_channels, kernel_size, stride, padding, normalize=True, bias=True,
                               weight_norm=None):
    conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
    if weight_norm:
        conv = weight_norm(conv)
    layers = [conv]
    if normalize:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    return layers


def conv1d_trans_bn_lrelu_block(in_channels, out_channels, kernel_size, stride, padding, alpha=0.2, normalize=True,
                                bias=True, weight_norm=None):
    conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
    if weight_norm:
        conv = weight_norm(conv)
    layers = [conv]
    if normalize:
        layers.append(nn.BatchNorm1d(out_channels))
    layers.append(nn.LeakyReLU(alpha, inplace=True))
    return layers


def conv2d_trans_bn_lrelu_block(in_channels, out_channels, kernel_size, stride, padding, alpha=0.2, normalize=True,
                                bias=True, weight_norm=None):
    conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
    if weight_norm:
        conv = weight_norm(conv)
    layers = [conv]
    if normalize:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(alpha, inplace=True))
    return layers


"""
Special Torch layer
"""


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class DynamicGNoise(nn.Module):
    def __init__(self, shape, std=0.05):
        super().__init__()
        if enable_cuda:
            self.noise = Variable(torch.zeros(1, *shape).cuda())
        else:
            self.noise = Variable(torch.zeros(1, *shape))
        self.std = std

    def forward(self, x):
        if not self.training: return x
        self.noise.data.normal_(0, std=self.std)

        return x + self.noise.expand(x.shape)


class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


"""
Layer utility functions
"""


def change_model_trainable(model: nn.Module, trainable=False):
    """ set a PyTorch module to trainable

    Args:
        model: model to set to trainable or not

    Returns:

    """
    for param in model.parameters():
        param.requires_grad = trainable


def freeze(model: nn.Module):
    change_model_trainable(model, False)


def unfreeze(model: nn.Module):
    change_model_trainable(model, True)
