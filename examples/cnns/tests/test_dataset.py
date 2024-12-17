'''
Author: Unknown
Date: 2024-04-05 01:04:14
LastEditTime: 2024-04-10 10:57:52
LastEditors: Unknown
Description: test whether the dataloader can be loaded correctly
    sample invocation: python tests/test_dataset.py --dataset ImageNet1k_imagefolder --datadir ./imagenet1k_imagefolder
FilePath: /Unknown/tests/test_dataset.py
'''
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print(sys.path)

# print('Start training...')

from datasets import get_cifar_dataloaders, get_mnist_dataloaders

# train_loader, test_loader = get_cifar_dataloaders(128, 128, 'cifar10', 4, datadir='/media/data/Unknown/cifar10', skip_download_check=True)
# train_loader, test_loader = get_cifar_dataloaders(128, 128, 'cifar100', 4, datadir='/media/data/Unknown/cifar100', skip_download_check=True)
# train_loader, test_loader = get_mnist_dataloaders(128, 128, 4, datadir='/media/data/Unknown/mnist')

import argparse

parser = argparse.ArgumentParser(description='PyTorch Dataloader Test')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--num_workers', default=4, type=int, help='num_workers')
parser.add_argument('--datadir', default='_dataset', type=str, help='dataset dir')

args = parser.parse_args()



# show some images
# cifar10
train_loader, test_loader = get_cifar_dataloaders(args.batch_size, args.batch_size, args.dataset, args.num_workers, datadir=args.datadir, skip_download_check=True)
print('Loading Done.')
iter_loader = iter(train_loader)
images, labels = next(iter_loader)
print(images.shape)
print(labels.shape)
# visualize
# import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from utils.image import recover_image
to_img = ToPILImage()
img = make_grid(images)
img = recover_image(img, args.dataset)
img = to_img(img)
# plt.imshow(np.asarray(img))
# plt.savefig(f'./{args.dataset}_sample.png')
# print(f'Save {args.dataset}_sample.png Done.')
img.save(f'./{args.dataset}_sample.png')
