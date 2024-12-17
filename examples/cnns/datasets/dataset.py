'''
Author: Unknown
Date: 2023-04-07 18:43:32
LastEditTime: 2024-04-10 10:16:24
LastEditors: Unknown
Description: dataset file from foresight/dataset.py, modified and update with more fns.
FilePath: /Unknown/datasets/dataset.py
'''

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, SVHN, ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import transforms

from .imagenet16 import *

# DATASET_SUPPORTED = ['cifar10', 'cifar100', 'svhn', 'ImageNet16-120', 'ImageNet1k']
DATASET_SUPPORTED = ['cifar10', 'cifar100', 'ImageNet1k', 'ImageNet1k_imagefolder'] # need to check std and mean for other datasets

def get_cifar_dataloaders(train_batch_size, test_batch_size, dataset, num_workers, resize=None, datadir='_dataset', skip_download_check=False, data_augmentation=True):
    """
    get cifar dataset train and test dataloaders.

    :param train_batch_size     is the train dataloader batch size
    :param test_batch_size      is the test dataloader batch size
    :param dataset              only support ['cifar10', 'cifar100', 'svhn', 'ImageNet16-120', 'ImageNet1k']
    :param num_workers          DataLodaer n_workers
    :param datadir              root dataset dir, storing all dataset data.
    :param skip_download_check  param to speed up process, skip dataset exist checking, and eliminate the logs
    
    :return (train_dataloader, test_dataloader)
    """
    if dataset not in DATASET_SUPPORTED:
        raise ValueError(f'Check dataset name! "{dataset}" not supported.')

    if 'ImageNet16' in dataset:
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std  = [x / 255 for x in [63.22,  61.26 , 65.09]]
        size, pad = 16, 2
    elif dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        # std = (0.2470, 0.2435, 0.2616)
        std = (0.2023, 0.1994, 0.2010)  # from AdamS code
        size, pad = 32, 4
    elif dataset == 'cifar100':
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2762)
        size, pad = 32, 4
    elif 'svhn' in dataset:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        size, pad = 32, 0
    elif dataset == 'ImageNet1k':
        from .h5py_dataset import H5Dataset
        size,pad = 224,2
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)
        #resize = 256
    elif dataset == 'ImageNet1k_imagefolder':
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)
        size, pad = 224, 2
        #resize = 256

    if resize is None:
        resize = size

    train_transform = transforms.Compose([
        transforms.RandomCrop(size, padding=pad),
        transforms.Resize(resize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
    ]) if data_augmentation else transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
    ])

    if dataset == 'cifar10':
        train_dataset = CIFAR10(datadir, True, train_transform, download=(not skip_download_check))
        test_dataset = CIFAR10(datadir, False, test_transform, download=(not skip_download_check))
    elif dataset == 'cifar100':
        train_dataset = CIFAR100(datadir, True, train_transform, download=(not skip_download_check))
        test_dataset = CIFAR100(datadir, False, test_transform, download=(not skip_download_check))
    elif dataset == 'svhn':
        train_dataset = SVHN(datadir, split='train', transform=train_transform, download=True)
        test_dataset = SVHN(datadir, split='test', transform=test_transform, download=True)
    elif dataset == 'ImageNet16-120':
        train_dataset = ImageNet16(os.path.join(datadir, 'ImageNet16-120'), True , train_transform, 120)
        test_dataset  = ImageNet16(os.path.join(datadir, 'ImageNet16-120'), False, test_transform , 120)
    elif dataset == 'ImageNet1k':
        train_dataset = H5Dataset(os.path.join(datadir, 'imagenet-train-256.h5'), transform=train_transform)
        test_dataset  = H5Dataset(os.path.join(datadir, 'imagenet-val-256.h5'),   transform=test_transform)
    elif dataset == 'ImageNet1k_imagefolder':
        train_dataset = ImageFolder(os.path.join(datadir, 'train'), train_transform)
        test_dataset  = ImageFolder(os.path.join(datadir, 'val'),   test_transform)
    else:
        raise ValueError('There are no more cifars or imagenets.')

    train_loader = DataLoader(
        train_dataset,
        train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)
    test_loader = DataLoader(
        test_dataset,
        test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)

    return train_loader, test_loader


def get_mnist_dataloaders(train_batch_size, val_batch_size, num_workers, datadir='_dataset'):

    data_transform = Compose([transforms.ToTensor()])

    # Normalise? transforms.Normalize((0.1307,), (0.3081,))

    train_dataset = MNIST(datadir, True, data_transform, download=True)
    test_dataset = MNIST(datadir, False, data_transform, download=True)

    train_loader = DataLoader(
        train_dataset,
        train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)
    test_loader = DataLoader(
        test_dataset,
        val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)

    return train_loader, test_loader
