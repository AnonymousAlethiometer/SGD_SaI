import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F


import einops
from datasets import get_cifar_dataloaders
from models import get_model
from maskers import get_masker

from lightning import seed_everything

seed_everything(42)


model = get_model("resnet18")
masker = get_masker("resnet18")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_loader, test_loader = get_cifar_dataloaders(train_batch_size=128, test_batch_size=64, dataset='cifar10', num_workers=4, datadir='/media/data/Unknown/cifar10', skip_download_check=True)
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

    loss = criterion(outputs, targets)
    print(loss)

    grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)
    
    # all_grads = torch.tensor([g.data.norm(2) for g in grads])
    all_grads = [g.norm(2) for g in grads]
    all_grads = torch.stack(all_grads)

    all_grads_stds = [g.std() for g in grads]
    all_grads_stds = torch.stack(all_grads_stds)

    all_grads_snr = [g.norm(2) / g.std() for g in grads]
    all_grads_snr = torch.stack(all_grads_snr)

    # # calculate std of gradients norm
    grad_std = torch.std(all_grads)  # calculate log of std of gradients norm
    log_grad_std = grad_std.log()

    mask_loss = log_grad_std

    new_grads = torch.autograd.grad(mask_loss, masker.parameters(), retain_graph=True, create_graph=True)

    mask_loss.backward()

    print(masker.conv1.weight.grad)
    print(new_grads[0])
