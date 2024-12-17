import torch
import torchvision
from torchvision import transforms


def denormalize(x, mean, std):
    x = x * std + mean
    return x

def recover_image(img, dataset):
    if 'ImageNet16' in dataset:
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std  = [x / 255 for x in [63.22,  61.26 , 65.09]]
        size, pad = 16, 2
    elif 'cifar' in dataset:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        size, pad = 32, 4
    elif 'svhn' in dataset:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        size, pad = 32, 0
    elif dataset == 'ImageNet1k':
        size,pad = 224,2
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)
        #resize = 256
    elif dataset == 'ImageNet1k_imagefolder':
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)
        size, pad = 224, 2
        #resize = 256
    
    inv_normalize = transforms.Normalize(
        mean = [-m/s for m, s in zip(mean, std)],
        std = [1/s for s in std]
    )

    inv_tensor = inv_normalize(img)
    return inv_tensor