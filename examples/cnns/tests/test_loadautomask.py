import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einops

import matplotlib.pyplot as plt
import numpy as np
from datasets.dataset import get_cifar_dataloaders

import torch
import torch.nn as nn
import torch.nn.functional as F


from models import get_model
from maskers import get_masker

from lightning import seed_everything

seed_everything(42)


model = get_model("resnet18")
masker = get_masker("resnet18")

ckpt_path = 'logs/cifar10/automasker/resnet18-automasker-simpletrain-withval/lightning_logs/version_0/checkpoints/epoch=199-step=156400.ckpt'
ckpt = torch.load(ckpt_path, map_location='cpu')

sd = ckpt['state_dict'].copy()
for k in list(sd.keys()):
    if k.startswith('model.'):
        # sd[k[len("model."):]] = sd[k]
        del sd[k]
    elif k.startswith('masker.'):
        sd[k[len("masker."):]] = sd[k]
        del sd[k]

masker.load_state_dict(sd)

print(masker)


train_loader, test_loader = get_cifar_dataloaders(train_batch_size=1, test_batch_size=64, dataset='cifar10', num_workers=4, datadir='/media/data/Unknown/cifar10', skip_download_check=True)
patch_num = 8


def mask_generator(masker, x):
    mask = masker(x)

    b, c, h, w = x.shape
    # repeat mask to match the shape of x
    rh = h // patch_num
    rw = w // patch_num
    mask = einops.repeat(mask, 'b (1 ph pw) -> b c (rh ph) (rw pw)', rh=rh, rw=rw, ph=patch_num, pw=patch_num, c=c, b=b)

    # rectify mask from (-inf, inf) to (0, 1)
    mask = F.sigmoid(mask)

    return mask


for batch_idx, (inputs, targets) in enumerate(train_loader):
    print(inputs.shape, targets.shape)

    mask = mask_generator(masker, inputs)
    print(mask.shape)

    inputs = inputs * mask
    print(inputs.shape)

    outputs = model(inputs)
    print(outputs.shape)


    # visualize mask
    mask = mask[0].permute(1, 2, 0).detach().numpy()
    print(mask)
    plt.imshow(mask, cmap='hot', interpolation='nearest')
    plt.savefig('mask.png')

    # wait for input
    input()
    
    # loss = criterion(outputs, targets)
    # print(loss)

    # grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)
    
    # # all_grads = torch.tensor([g.data.norm(2) for g in grads])
    # all_grads = [g.norm(2) for g in grads]
    # all_grads = torch.stack(all_grads)

    # # # calculate std of gradients norm
    # grad_std = torch.std(all_grads)  # calculate log of std of gradients norm
    # log_grad_std = grad_std.log()

    # mask_loss = log_grad_std

    # new_grads = torch.autograd.grad(mask_loss, masker.parameters(), retain_graph=True, create_graph=True)

    # mask_loss.backward()

    # print(masker.conv1.weight.grad)
    # print(new_grads[0])




