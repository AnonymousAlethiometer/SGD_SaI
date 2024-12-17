'''
Author: Unknown
Date: 2024-02-16 23:45:57
LastEditTime: 2024-05-15 00:26:08
LastEditors: Unknown
Description: 
FilePath: /Unknown/utils/lr_scheduler.py
'''
import torch.optim as optim


def lr_schedulers(optimizer, sch_name, **kwargs):
    if sch_name == 'StepLR':
        return optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif sch_name == 'MultiStepLR':
        return optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
    elif sch_name == 'ExponentialLR':
        return optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif sch_name == 'CosineAnnealingLR':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif sch_name == 'ReduceLROnPlateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    elif sch_name == 'CyclicLR':
        return optim.lr_scheduler.CyclicLR(optimizer, **kwargs)
    elif sch_name == 'OneCycleLR':
        return optim.lr_scheduler.OneCycleLR(optimizer, **kwargs)
    elif sch_name == 'CosineAnnealingWarmRestarts':
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **kwargs)
    elif sch_name == 'SWDLambdaLR' or sch_name == 'SWDLambdaLR_cifar10':
        lambda_lr = lambda epoch: 0.1 ** (epoch // 80) 
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    elif sch_name == 'SWDLambdaLR_cifar100':
        # lambda_lr = lambda epoch: 0.1 ** (epoch // 80) 
        lambda_lr = lambda epoch: 1 if epoch < 100 else 0.1 if epoch < 150 else 0.01
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    else:
        raise('Unspecified lr_scheduler.')
