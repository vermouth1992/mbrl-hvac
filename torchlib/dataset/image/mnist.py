"""
MNIST dataset utils
"""

import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

from .. import dataset_root_path
from ...common import enable_cuda

_default_mnist_dataset = None


def get_mnist_default_transform():
    """ The default transform come with data augmentation """
    return transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def get_mnist_dataset(train, transform=None):
    if transform is None:
        transform = get_mnist_default_transform()
        global _default_mnist_dataset
        if _default_mnist_dataset is None:
            _default_mnist_dataset = datasets.MNIST(dataset_root_path, train=train, download=True,
                                                    transform=transform)
        return _default_mnist_dataset
    else:
        return datasets.MNIST(dataset_root_path, train=train, download=True,
                              transform=transform)


def get_mnist_data_loader(train, batch_size=128, transform=None):
    kwargs = {'num_workers': 1, 'pin_memory': True} if enable_cuda else {}
    dataset = get_mnist_dataset(train, transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, **kwargs)
    return train_loader


def get_mnist_subset_data_loader(train, batch_size=128, transform=None, fraction=0.1):
    kwargs = {'num_workers': 1, 'pin_memory': True} if enable_cuda else {}
    dataset = get_mnist_dataset(train, transform)
    if type(fraction) == float:
        end = int(len(dataset) * fraction)
    elif type(fraction) == int:
        end = fraction
    else:
        raise ValueError('fraction has to be float or int')
    sampler = SubsetRandomSampler(list(range(0, end)))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size, sampler=sampler, shuffle=False, **kwargs)
    return train_loader


def get_mnist_raw_data():
    import keras
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    return (x_train, y_train), (x_test, y_test)
