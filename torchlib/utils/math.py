"""
Utilities for math operations
"""

from torchlib.common import eps


def normalize(x, mean, std):
    return (x - mean) / (std + eps)


def unnormalize(x, mean, std):
    return x * std + mean


def log_to_log2(x):
    return x * 1.4426950408889634
