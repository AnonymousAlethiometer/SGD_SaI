from __future__ import print_function
'''
Author: Unknown
Date: 2024-03-01 12:08:39
LastEditTime: 2024-03-02 18:44:49
LastEditors: Unknown
Description: 
    copied from https://github.com/wenwei202/pytorch-examples/blob/autogrow/cifar10/get_mean_std.py
    reference https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
FilePath: /Unknown/tests/get_mean_std.py
'''
'''Get dataset mean and std with PyTorch.'''

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from copy import deepcopy
import re

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

# from models import *
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset')
parser.add_argument('--batch_size', default='200', type=int, help='dataset')
parser.add_argument('--corr', default='1.0', type=float, help='whether correct the std')

args = parser.parse_args()

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
])

if 'SVHN' == args.dataset:
    trainset = getattr(torchvision.datasets, args.dataset)(root='./data/' + args.dataset, split='train', download=True,
                                                           transform=transform_train)
else:
    trainset = getattr(torchvision.datasets, args.dataset)(root='./data/'+args.dataset, train=True, download=True, transform=transform_train)
print('%d training samples.' % len(trainset))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=2)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
h, w = 0, 0
for batch_idx, (inputs, targets) in enumerate(trainloader):
    inputs = inputs.to(device)
    if batch_idx == 0:
        h, w = inputs.size(2), inputs.size(3)
        print(inputs.min(), inputs.max())
        chsum = inputs.sum(dim=(0, 2, 3), keepdim=True)
    else:
        chsum += inputs.sum(dim=(0, 2, 3), keepdim=True)
mean = chsum/len(trainset)/h/w
print("--------------------")
print('mean: %s' % mean.view(-1).numpy())

chsum = None
for batch_idx, (inputs, targets) in enumerate(trainloader):
    inputs = inputs.to(device)
    if batch_idx == 0:
        chsum = (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
    else:
        chsum += (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
# std = torch.sqrt(chsum/(len(trainset) * h * w - 1))
std = torch.sqrt(chsum/(len(trainset) * h * w - args.corr))
print('std: %s' % std.view(-1).numpy())

print("--------------------")

print('Done!')

# cifar 10 mean and std
# coplit results
# mean: [0.49139968 0.48215841 0.44653091]
# std: [0.24703223 0.24348513 0.26158784]
# mine results
# mean: [0.49139968 0.4821584  0.44653103]
# std: [0.24703227 0.24348514 0.26158792]


# cifar 100 mean and std
# coplit results
# mean: [0.50707516 0.48654887 0.44091784]
# std: [0.26733429 0.25643846 0.27615047]
# mine results
# mean: [0.5070751  0.486549   0.44091788]
# std: [0.26733428 0.25643846 0.27615044]
