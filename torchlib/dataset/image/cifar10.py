"""
Cifar10 dataloader
"""

import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

from torchlib.common import enable_cuda
from torchlib.dataset import dataset_root_path


def get_cifar10_default_transform(train, augmentation=False):
    """ The default transform come with data augmentation """
    if train and augmentation:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])


def get_cifar10_dataset(train, transform=None, augmentation=False):
    if transform is None:
        transform = get_cifar10_default_transform(train, augmentation)
    return datasets.CIFAR10(dataset_root_path, train=train, download=False, transform=transform)


def get_cifar10_data_loader(train, batch_size=128, transform=None, augmentation=False):
    kwargs = {'num_workers': 0, 'pin_memory': True} if enable_cuda else {}
    dataset = get_cifar10_dataset(train, transform, augmentation)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, **kwargs)
    return train_loader


def get_cifar10_subset_data_loader(train, batch_size=128, transform=None, fraction=0.1):
    kwargs = {'num_workers': 1, 'pin_memory': True} if enable_cuda else {}
    dataset = get_cifar10_dataset(train, transform, augmentation=False)
    if type(fraction) == float:
        end = int(len(dataset) * fraction)
    elif type(fraction) == int:
        end = fraction
    else:
        raise ValueError('fraction has to be float or int')
    sampler = SubsetRandomSampler(list(range(0, end)))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size, sampler=sampler, shuffle=False, **kwargs)
    return train_loader


def get_cifar10_raw_data():
    import keras
    return keras.datasets.cifar10.load_data()
