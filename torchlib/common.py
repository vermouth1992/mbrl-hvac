# check cuda for torch
import numpy as np
import torch

from . import enable_cuda

__all__ = ['enable_cuda', 'FloatTensor', 'LongTensor', 'map_location', 'eps']

FloatTensor = torch.cuda.FloatTensor if enable_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if enable_cuda else torch.LongTensor
map_location = None if enable_cuda else 'cpu'
eps = np.finfo(np.float32).eps.item()

if enable_cuda:
    device = 'cuda:0'
else:
    device = 'cpu'


def convert_numpy_to_tensor(numpy_array, location='gpu'):
    if isinstance(numpy_array, np.ndarray):
        tensor = torch.from_numpy(numpy_array)
        if enable_cuda and location == 'gpu':
            return tensor.cuda()
        else:
            return tensor
    elif isinstance(numpy_array, list) or isinstance(numpy_array, tuple):
        out = []
        for array in numpy_array:
            tensor = torch.from_numpy(array)
            if enable_cuda and location == 'gpu':
                out.append(tensor.cuda())
            else:
                out.append(tensor)
        return out

    else:
        raise ValueError('Unknown numpy array data type {}'.format(type(numpy_array)))


def move_tensor_to_gpu(tensors):
    if not enable_cuda:
        return tensors
    else:
        if isinstance(tensors, list) or isinstance(tensors, tuple):
            out = []
            for tensor in tensors:
                out.append(tensor.cuda())
            return out

        else:
            return tensors.cuda()
