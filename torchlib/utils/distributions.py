"""
Rewrite Pytorch builtin distribution function to favor policy gradient
1. Normal distribution with multiple mean and std as a single distribution
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Independent, Transform, constraints, TransformedDistribution, Distribution
from torch.distributions import Normal, Beta, AffineTransform

from torchlib.common import eps


class TanhTransform(Transform):
    domain = constraints.real
    codomain = constraints.interval(-1., 1.)
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        return 0.5 * (torch.log1p(y + eps) - torch.log1p(-y + eps))

    def log_abs_det_jacobian(self, x, y):
        return np.log(4.) + 2. * x - 2 * F.softplus(2. * x)


class TanhNormal(TransformedDistribution):
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        base_dist = Normal(loc, scale)
        super(TanhNormal, self).__init__(base_dist, TanhTransform(cache_size=1), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(TanhNormal, _instance)
        return super(TanhNormal, self).expand(batch_shape, _instance=new)

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.scale


class IndependentNormal(Distribution):
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        self.base_dist = Independent(Normal(loc=loc, scale=scale, validate_args=validate_args), len(loc.shape) - 1,
                                     validate_args=validate_args)
        super(IndependentNormal, self).__init__(self.base_dist.batch_shape, self.base_dist.event_shape,
                                                validate_args=validate_args)

    def log_prob(self, value):
        return self.base_dist.log_prob(value)

    @property
    def mean(self):
        return self.base_dist.mean

    @property
    def variance(self):
        return self.base_dist.variance

    def sample(self, sample_shape=torch.Size()):
        return self.base_dist.sample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self.base_dist.rsample(sample_shape)

    def entropy(self):
        entropy = self.base_dist.entropy()
        return entropy


class IndependentTanhNormal(Distribution):
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        self.base_dist = Independent(TanhNormal(loc=loc, scale=scale, validate_args=validate_args), len(loc.shape) - 1,
                                     validate_args=validate_args)
        super(IndependentTanhNormal, self).__init__(self.base_dist.batch_shape, self.base_dist.event_shape,
                                                    validate_args=validate_args)

    def log_prob(self, value):
        return self.base_dist.log_prob(value)

    @property
    def mean(self):
        return self.base_dist.mean

    @property
    def variance(self):
        return self.base_dist.variance

    def sample(self, sample_shape=torch.Size()):
        return self.base_dist.sample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self.base_dist.rsample(sample_shape)

    def entropy(self):
        entropy = self.base_dist.entropy()
        return entropy


class RescaledBeta(TransformedDistribution):
    arg_constraints = {'concentration1': constraints.positive, 'concentration0': constraints.positive}
    support = constraints.interval(-1., 1.)
    has_rsample = True

    def __init__(self, concentration1, concentration0, validate_args=None):
        base_distribution = Beta(concentration1, concentration0, validate_args=validate_args)
        super(RescaledBeta, self).__init__(base_distribution=base_distribution,
                                           transforms=AffineTransform(loc=-1., scale=2.))

    def entropy(self):
        return self.base_dist.entropy() + math.log(2.)

    def sample(self, sample_shape=torch.Size()):
        out = super(RescaledBeta, self).sample(sample_shape)
        return torch.clamp(out, -1. + eps, 1. - eps)

    def rsample(self, sample_shape=torch.Size()):
        out = super(RescaledBeta, self).rsample(sample_shape)
        return torch.clamp(out, -1. + eps, 1. - eps)


class IndependentRescaledBeta(Distribution):
    arg_constraints = {'concentration1': constraints.positive, 'concentration0': constraints.positive}
    support = constraints.interval(-1., 1.)
    has_rsample = True

    def __init__(self, concentration1, concentration0, validate_args=None):
        self.base_dist = Independent(RescaledBeta(concentration1, concentration0, validate_args),
                                     len(concentration1.shape) - 1, validate_args=validate_args)
        super(IndependentRescaledBeta, self).__init__(self.base_dist.batch_shape, self.base_dist.event_shape,
                                                      validate_args=validate_args)

    def log_prob(self, value):
        return self.base_dist.log_prob(value)

    @property
    def mean(self):
        return self.base_dist.mean

    @property
    def variance(self):
        return self.base_dist.variance

    def sample(self, sample_shape=torch.Size()):
        return self.base_dist.sample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self.base_dist.rsample(sample_shape)

    def entropy(self):
        entropy = self.base_dist.entropy()
        return entropy
