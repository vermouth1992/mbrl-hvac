"""
A sampler defines a method to sample random data from certain distribution.
"""

from typing import List

import numpy as np


class BaseSampler(object):
    def __init__(self):
        pass

    def sample(self, shape, *args):
        raise NotImplementedError


class IntSampler(BaseSampler):
    def __init__(self, low, high=None):
        super(IntSampler, self).__init__()
        if high is None:
            self.low = 0
            self.high = low
        else:
            self.low = low
            self.high = high

    def sample(self, shape, *args):
        return np.random.randint(low=self.low, high=self.high, size=shape, dtype=np.int64)


class UniformSampler(BaseSampler):
    def __init__(self, low, high):
        super(UniformSampler, self).__init__()
        self.low = np.array(low)
        self.high = np.array(high)

        assert self.low.shape == self.high.shape, 'The shape of low and high must be the same. Got low type {} and high type {}'.format(
            self.low.shape, self.high.shape)

    def sample(self, shape, *args):
        return np.random.uniform(low=self.low, high=self.high, size=shape + self.low.shape).astype(np.float32)


class GaussianSampler(BaseSampler):
    def __init__(self, mu=0.0, sigma=1.0):
        super(GaussianSampler, self).__init__()
        self.mu = mu
        self.sigma = sigma

    def sample(self, shape, *args):
        return np.random.normal(self.mu, self.sigma, shape)


class GaussianMixtureSampler(BaseSampler):
    """ Sample from GMM with prior probability distribution """

    def __init__(self, mu: List, sigma: List, prob=None):
        assert type(mu) == list and type(sigma) == list, 'mu and sigma must be list'
        assert len(mu) == len(sigma), 'length of mu and sigma must be the same'
        if type(prob) == list:
            assert len(mu) == len(prob) and np.sum(prob) == 1., 'The sum of probability list should be 1.'
        super(GaussianMixtureSampler, self).__init__()
        self.mu = mu
        self.sigma = sigma
        self.prob = prob

    def sample(self, shape, *args):
        ind = np.random.choice(len(self.mu), p=self.prob)
        return np.random.randn(*shape) * self.sigma[ind] + self.mu[ind]


class ConditionGaussianSampler(BaseSampler):
    """ Conditional Gaussian sampler """

    def __init__(self, mu: List, sigma: List):
        assert type(mu) == list and type(sigma) == list, 'mu and sigma must be list'
        assert len(mu) == len(sigma), 'length of mu and sigma must be the same'
        super(ConditionGaussianSampler, self).__init__()
        self.mu = np.expand_dims(np.array(mu), axis=1)
        self.sigma = np.expand_dims(np.array(sigma), axis=1)

    def sample(self, shape, *args):
        ind = args[0]
        return np.random.randn(*shape) * self.sigma[ind] + self.mu[ind]
