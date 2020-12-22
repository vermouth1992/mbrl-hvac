import numpy as np
import torch
from torch.utils.data import TensorDataset, Dataset
from torchlib.common import enable_cuda


def create_tensor_dataset(data):
    tensor_data = []
    for d in data:
        if isinstance(d, np.ndarray):
            tensor_data.append(torch.from_numpy(d))
        elif isinstance(d, torch.Tensor):
            tensor_data.append(d)
        else:
            raise ValueError('Unknown data type {}'.format(type(d)))
    return TensorDataset(*tensor_data)


class TupleDataset(Dataset):
    """
    A tuple of tensordataset. Used for (input), (target)
    """

    def __init__(self, tuples_tensors):
        """

        Args:
            source: a tuple of a list of numpy array
            target: a tuple of a list of numpy array
        """
        self.tensor_datasets = []
        for data in tuples_tensors:
            assert isinstance(data, tuple), \
                'Each element in tuples_tensors must also be a tuple. Got {}'.format(type(data))
            self.tensor_datasets.append(create_tensor_dataset(data))

        assert all(len(self.tensor_datasets[0]) == len(dataset) for dataset in self.tensor_datasets)

    def __getitem__(self, index):
        return tuple(dataset[index] for dataset in self.tensor_datasets)

    def __len__(self):
        return len(self.tensor_datasets[0])


def create_data_loader(data, batch_size=32, shuffle=True, drop_last=False):
    """ Create a data loader given numpy array x and y

    Args:
        data: a tuple (x, y, z, ...) where they have common first shape dim.

    Returns: Pytorch data loader

    """
    kwargs = {'num_workers': 0, 'pin_memory': True} if enable_cuda else {}
    dataset = create_tensor_dataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last, **kwargs)
    return loader


def create_tuple_data_loader(tuples_data, batch_size=32, shuffle=True, drop_last=False):
    kwargs = {'num_workers': 0, 'pin_memory': True} if enable_cuda else {}
    dataset = TupleDataset(tuples_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last, **kwargs)
    return loader
