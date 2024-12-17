'''
Author: Unknown
Date: 2024-03-03 14:48:58
LastEditTime: 2024-05-17 20:40:30
LastEditors: Unknown
Description: 
FilePath: /Unknown/utils/train_utils.py
'''

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils.optimizer import optimizers
from utils.lr_scheduler import lr_schedulers
from utils.shared_variables import SHARED_VARS
from datasets import get_cifar_dataloaders, get_mnist_dataloaders
from models import get_model
import argparse
from torchattacks import PGD, PGDL2
# from lightning import seed_everything
import yaml
import time
from torcheval.metrics import MulticlassAccuracy
from accelerate import Accelerator
import copy


def basic_setup():
    parser = argparse.ArgumentParser(description='PyTorch Baseline Training')
    parser.add_argument('--model', default='resnet50', type=str, help='model name')
    # dataset settings
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
    parser.add_argument('--data_dir', default='./', type=str, help='dataset root directory')
    parser.add_argument('--data_aug', action='store_true', help='whether to use data augmentation')
    parser.add_argument('--num_workers', default=8, type=int, help='dataloader num_workers')
    # training settings
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--gpu', default='0', type=str, help='gpu id')
    parser.add_argument('--batch_size', default=128, type=int, help='train batch size')
    parser.add_argument('--test_batch_size', default=128, type=int, help='test batch size')
    parser.add_argument('--epochs', default=200, type=int, help='train epochs')
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint path')
    parser.add_argument('--save_path', default='checkpoints', type=str, help='save checkpoint path')
    parser.add_argument('--early_stop', action='store_true', help='whether to use early stop')
    parser.add_argument('--grad_clip_norm', default=0, type=float, help='for normalizing gradients to a certain value')
    parser.add_argument('--grad_clip_value', default=0, type=float, help='to clip gradients to a minimum and maximum value')
    parser.add_argument('--warmup_steps', default=0, type=int, help='warmup steps')
    # lr settings
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_scheduler', default='none', type=str, help='learning rate scheduler (cosine, cosine_restart, step, exp, none, fixed)')
    # optimizer settings
    parser.add_argument('--optimizer', default='sgd', type=str, help='optimizer (sgd, adam, adamw, rmsprop)')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='SGD/Adam/AdamW weight decay')
    parser.add_argument('--wd_ratio_monitor', action='store_true', help='whether to monitor the weight decay rate')
    # adversarial attack
    parser.add_argument('--adv_types', nargs='+', default=['none'], type=str, help='adversarial attack types, e.g., pgd_linf, pgd_l2')
    parser.add_argument('--adv_last_nepochs', default=5, type=int, help='last n epochs to attack')
    parser.add_argument('--adv_train', action='store_true', help='whether to use adversarial training')
    parser.add_argument('--adv_train_eps', default=8/255, type=float, help='epsilon of attack during training')
    parser.add_argument('--adv_train_alpha', default=2/255, type=float, help='itertion number of attack during training')
    parser.add_argument('--adv_train_iters', default=10, type=int, help='itertion number of attack during training')
    parser.add_argument('--adv_train_randinit', action='store_false', help='whether to use random initialization during training (default: on)')
    parser.add_argument('--adv_test', action='store_true', help='whether to use adversarial test')
    parser.add_argument('--adv_test_eps', default=8/255, type=float, help='epsilon of attack during testing')
    parser.add_argument('--adv_test_alpha', default=2/255, type=float, help='itertion number of attack during testing')
    parser.add_argument('--adv_test_iters', default=20, type=int, help='itertion number of attack during testing')
    parser.add_argument('--adv_test_randinit', action='store_false', help='whether to use random initialization during testing (default: on)')
    # logging settings
    parser.add_argument('--log_wandb', action='store_true', help='whether to log to wandb')


    args = parser.parse_args()
    print("get args done!")


    print('Freezing the seed...')
    # set random seed for reproducibility 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    # seed_everything(args.seed, workers=True)

    print('Setting deterministic cudnn...')
    # set deterministic cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # check whether gpu exits, and whether to use gpu
    print('Checking GPU...')
    print('-'*25)
    gpu_exist = False
    try:
        import subprocess
        subprocess.check_output('nvidia-smi')
        print('Nvidia GPU found!')
        gpu_exist = True
    except Exception: # this command not being found can raise quite a few different errors depending on the configuration
        print('No Nvidia GPU in system!')
    print(f'GPU exists: {gpu_exist}, GPU used: {torch.cuda.is_available()}')

    # set gpu id
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # set device
    _device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Assign to GPU: {args.gpu}')
    print(f'CUDA_VISIBLE_DEVICES: [{os.environ["CUDA_VISIBLE_DEVICES"]}]')
    print(f'Using device: {_device}')
    print('-'*25)


    # set dataloader
    print('Loading dataset...')
    s0 = time.time()
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    if args.dataset == 'cifar10':
        train_loader, test_loader = get_cifar_dataloaders(args.batch_size, args.test_batch_size, 'cifar10', args.num_workers, datadir=dataset_dir, skip_download_check=False, data_augmentation=args.data_aug)
        args.n_classes, args.input_size = 10, 32
    elif args.dataset == 'cifar100':
        train_loader, test_loader = get_cifar_dataloaders(args.batch_size, args.test_batch_size, 'cifar100', args.num_workers, datadir=dataset_dir, skip_download_check=False, data_augmentation=args.data_aug)
        args.n_classes, args.input_size = 100, 32
    elif args.dataset == 'mnist':
        train_loader, test_loader = get_mnist_dataloaders(args.batch_size, args.test_batch_size, args.num_workers, datadir=dataset_dir)
        args.n_classes, args.input_size = 10, 28
    elif args.dataset == 'ImageNet1k':
        train_loader, test_loader = get_cifar_dataloaders(args.batch_size, args.test_batch_size, 'ImageNet1k', args.num_workers, datadir=dataset_dir, skip_download_check=False, data_augmentation=args.data_aug)
        args.n_classes, args.input_size = 1000, 224
    elif args.dataset == 'ImageNet1k_imagefolder':
        train_loader, test_loader = get_cifar_dataloaders(args.batch_size, args.test_batch_size, 'ImageNet1k_imagefolder', args.num_workers, datadir=dataset_dir, skip_download_check=False, data_augmentation=args.data_aug)
        args.n_classes, args.input_size = 1000, 224
    else:
        raise ValueError('Dataset not supported')
    print(f'Loading dataset done. Time: {time.time()-s0:.2f}s')

    # set model
    _model_params = {  # model parameters
        'num_classes': args.n_classes
    }
    if args.model.startswith('vit'):
        _model_params['image_size'] = args.input_size # manually set image_size for ViT
    _model = get_model(args.model, **_model_params) # get model from models/__init__.py
    _model = _model.to(_device)

    # set optimizer
    _params = _model.parameters()
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(_params, lr=args.lr)
    else:
        optimizer = None
        optimizer = optimizers(_model, args.optimizer, args.lr, args.weight_decay)
        print(f'Using SWD optimizer: {args.optimizer}')

    # set lr scheduler
    if args.lr_scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.5), int(args.epochs*0.75)], gamma=0.1)  # refer to paper: benchopt
    elif args.lr_scheduler == 'exp':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    elif args.lr_scheduler == 'cosine_restart':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=80, T_mult=1)
    elif args.lr_scheduler == 'none' or args.lr_scheduler is None or args.lr_scheduler == 'fixed':
        lr_scheduler = None
    else:
        lr_scheduler = None
        # raise ValueError('lr_scheduler must be one of [cosine, step]')
        lr_scheduler = lr_schedulers(optimizer, args.lr_scheduler)
        print(f'Using SWD lr_scheduler: {args.lr_scheduler}')

    # set criterion
    criterion = nn.CrossEntropyLoss(reduction='mean')

    # set multiclass accuracy metrics
    train_acc = MulticlassAccuracy(num_classes=args.n_classes, device=_device)
    test_acc = MulticlassAccuracy(num_classes=args.n_classes, device=_device)

    # set adversarial attack
    adversary = {
        'train': {},
        'test': {}
    }
    adversary_metrics = {
        'train': {},
        'test': {}
    }
    for adv_type in args.adv_types:
        if adv_type not in ['none', 'pgd_linf', 'pgd_l2']:
            print(f'Adversarial attack type {adv_type} not supported')
            args.adv_types.remove(adv_type)
        
        if adv_type == 'pgd_linf':
            if args.adv_train:
                adversary_train = PGD(_model, eps=args.adv_train_eps, alpha=args.adv_train_alpha, steps=args.adv_train_iters, random_start=args.adv_train_randinit)
                adversary_train.set_device(_device)  # manually set device, since current _model is not on device
                adversary['train'][adv_type] = adversary_train
                adversary_metrics['train'][adv_type] = MulticlassAccuracy(num_classes=args.n_classes, device=_device)
            if args.adv_test:
                adversary_test = PGD(_model, eps=args.adv_test_eps, alpha=args.adv_test_alpha, steps=args.adv_test_iters, random_start=args.adv_test_randinit)
                adversary_test.set_device(_device)  # manually set device, since current _model is not on device
                adversary['test'][adv_type] = adversary_test
                adversary_metrics['test'][adv_type] = MulticlassAccuracy(num_classes=args.n_classes, device=_device)
        elif adv_type == 'pgd_l2':
            if args.adv_train:
                adversary_train = PGDL2(_model, eps=args.adv_train_eps, alpha=args.adv_train_alpha, steps=args.adv_train_iters, random_start=args.adv_train_randinit)
                adversary_train.set_device(_device)  # manually set device, since current _model is not on device
                adversary['train'][adv_type] = adversary_train
                adversary_metrics['train'][adv_type] = MulticlassAccuracy(num_classes=args.n_classes, device=_device)
            if args.adv_test:
                adversary_test = PGDL2(_model, eps=args.adv_test_eps, alpha=args.adv_test_alpha, steps=args.adv_test_iters, random_start=args.adv_test_randinit)
                adversary_test.set_device(_device)  # manually set device, since current _model is not on device
                adversary['test'][adv_type] = adversary_test
                adversary_metrics['test'][adv_type] = MulticlassAccuracy(num_classes=args.n_classes, device=_device)
        else:
            continue

    # create the root save directory if not exists
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # make logs dir, and version_n dir within the root save directory if not exists
    if not os.path.exists(os.path.join(args.save_path, 'logs')):
        version = 0
    else:
        # version = len([n for n in os.listdir(os.path.join(args.save_path, 'logs')) if n.startswith('version_') and os.path.isdir(os.path.join(args.save_path, 'logs', n))])
        last_idx = -1
        for n in os.listdir(os.path.join(args.save_path, 'logs')):
            if n.startswith('version_') and os.path.isdir(os.path.join(args.save_path, 'logs', n)):
                idx = int(n.split('_')[-1])
                last_idx = max(last_idx, idx)
        version = last_idx + 1
    args.root_save_path = os.path.join(args.save_path, 'logs', f'version_{version}')
    os.makedirs(args.root_save_path)
    os.makedirs(os.path.join(args.root_save_path, 'checkpoints'))

    # record configs as xml
    # with open(os.path.join(args.save_path, 'config.xml'), 'w') as f:
    #     f.write('<config>\n')
    #     for k, v in vars(args).items():
    #         f.write('\t<%s>%s</%s>\n' % (k, v, k))
    #     f.write('</config>\n')
    # record configs as yaml
    with open(os.path.join(args.root_save_path, 'config.yaml'), 'w') as f:
        # _content = {'args': vars(args)}
        _content = {'args': vars(copy.copy(args))}
        yaml.dump(_content, f)
    
    # set shared variables
    SHARED_VARS['args'] = args
    
    # return args, _device, _model, train_loader, test_loader, optimizer, lr_scheduler, criterion, adversary
    rt_dict = {
        'device': _device,
        'model': _model,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler,
        'criterion': criterion,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'adversary': adversary,
        'adversary_metrics': adversary_metrics,
    }
    
    print_stats(_model, criterion, optimizer, lr_scheduler)

    return args, rt_dict

def print_stats(model, criterion, optimizer, lr_scheduler):
    '''
    | Name      | Type               | Params
    -------------------------------------------------
    0 | model     | ResNet             | 11.2 M
    1 | criterion | CrossEntropyLoss   | 0     
    2 | train_acc | MulticlassAccuracy | 0     
    3 | test_acc  | MulticlassAccuracy | 0     
    -------------------------------------------------
    11.2 M    Trainable params
    0         Non-trainable params
    11.2 M    Total params
    44.696    Total estimated model params size (MB)
    '''
    print(f'  | Name         | {"Type":20s} | Params')
    print('------------------------------------------------------')
    print(f'0 | model        | {type(model).__name__:20s} | {sum(p.numel() for p in model.parameters())/1e6:.1f} M')
    print(f'1 | criterion    | {type(criterion).__name__:20s} | 0     ')
    print(f'2 | optimizer    | {type(optimizer).__name__:20s} | 0     ')
    print(f'3 | lr_scheduler | {type(lr_scheduler).__name__:20s} | 0     ')
    print('------------------------------------------------------')
    print(f'{sum(p.numel() for p in model.parameters())/1e6:.1f} M    Trainable params')
    print(f'0         Non-trainable params')
    print(f'{sum(p.numel() for p in model.parameters())/1e6:.1f} M    Total params')
    print(f'{sum(p.numel() for p in model.parameters())*4/1e6:.3f}    Total estimated model params size (MB)')
    return


def basic_setup_distributed_hf_accelerate():
    parser = argparse.ArgumentParser(description='PyTorch Baseline Training')
    parser.add_argument('--model', default='resnet50', type=str, help='model name')
    # dataset settings
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
    parser.add_argument('--data_dir', default='./', type=str, help='dataset root directory')
    parser.add_argument('--data_aug', action='store_true', help='whether to use data augmentation')
    parser.add_argument('--num_workers', default=8, type=int, help='dataloader num_workers')
    # training settings
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--gpu', default='0', type=str, help='gpu id')
    parser.add_argument('--batch_size', default=128, type=int, help='train batch size')
    parser.add_argument('--test_batch_size', default=128, type=int, help='test batch size')
    parser.add_argument('--epochs', default=200, type=int, help='train epochs')
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint path')
    parser.add_argument('--save_path', default='checkpoints', type=str, help='save checkpoint path')
    parser.add_argument('--early_stop', action='store_true', help='whether to use early stop')
    parser.add_argument('--grad_clip_norm', default=0, type=float, help='for normalizing gradients to a certain value')
    parser.add_argument('--grad_clip_value', default=0, type=float, help='to clip gradients to a minimum and maximum value')
    parser.add_argument('--warmup_steps', default=0, type=int, help='warmup steps')
    parser.add_argument('--distributed', action='store_true', help='whether to use distributed training')
    parser.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--checkpointing_steps", type=str, default=None, help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="If the training should continue from a checkpoint folder.")
    # lr settings
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_scheduler', default='none', type=str, help='learning rate scheduler (cosine, cosine_restart, step, exp, none, fixed)')
    # optimizer settings
    parser.add_argument('--optimizer', default='sgd', type=str, help='optimizer (sgd, adam, adamw, rmsprop)')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='SGD/Adam/AdamW weight decay')
    parser.add_argument('--wd_ratio_monitor', action='store_true', help='whether to monitor the weight decay rate')
    # adversarial attack
    parser.add_argument('--adv_types', nargs='+', default=['none'], type=str, help='adversarial attack types, e.g., pgd_linf, pgd_l2')
    parser.add_argument('--adv_last_nepochs', default=5, type=int, help='last n epochs to attack')
    parser.add_argument('--adv_train', action='store_true', help='whether to use adversarial training')
    parser.add_argument('--adv_train_eps', default=8/255, type=float, help='epsilon of attack during training')
    parser.add_argument('--adv_train_alpha', default=2/255, type=float, help='itertion number of attack during training')
    parser.add_argument('--adv_train_iters', default=10, type=int, help='itertion number of attack during training')
    parser.add_argument('--adv_train_randinit', action='store_false', help='whether to use random initialization during training (default: on)')
    parser.add_argument('--adv_test', action='store_true', help='whether to use adversarial test')
    parser.add_argument('--adv_test_eps', default=8/255, type=float, help='epsilon of attack during testing')
    parser.add_argument('--adv_test_alpha', default=2/255, type=float, help='itertion number of attack during testing')
    parser.add_argument('--adv_test_iters', default=20, type=int, help='itertion number of attack during testing')
    parser.add_argument('--adv_test_randinit', action='store_false', help='whether to use random initialization during testing (default: on)')
    # logging settings
    parser.add_argument('--log_wandb', action='store_true', help='whether to log to wandb')

    args = parser.parse_args()


    # set random seed for reproducibility 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    # seed_everything(args.seed, workers=True)

    # set deterministic cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # check whether gpu exits, and whether to use gpu
    print('Checking GPU...')
    print('-'*25)
    gpu_exist = False
    try:
        import subprocess
        subprocess.check_output('nvidia-smi')
        print('Nvidia GPU found!')
        gpu_exist = True
    except Exception: # this command not being found can raise quite a few different errors depending on the configuration
        print('No Nvidia GPU in system!')
    print(f'GPU exists: {gpu_exist}, GPU used: {torch.cuda.is_available()}')

    if not args.distributed:
        # set gpu id
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        # set device
        _device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        print(f'Assign to GPU: {args.gpu}')
        print(f'CUDA_VISIBLE_DEVICES: [{os.environ["CUDA_VISIBLE_DEVICES"]}]')
        print(f'Using device: {_device}')
        print('-'*25)
    else:
        if args.fp16:
            accelerator = Accelerator(mixed_precision=args.mixed_precision)
        else:
            accelerator = Accelerator()
        _device = accelerator.device

    # set dataloader
    print('Loading dataset...')
    s0 = time.time()
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    if args.dataset == 'cifar10':
        train_loader, test_loader = get_cifar_dataloaders(args.batch_size, args.test_batch_size, 'cifar10', args.num_workers, datadir=dataset_dir, skip_download_check=False, data_augmentation=args.data_aug)
        args.n_classes, args.input_size = 10, 32
    elif args.dataset == 'cifar100':
        train_loader, test_loader = get_cifar_dataloaders(args.batch_size, args.test_batch_size, 'cifar100', args.num_workers, datadir=dataset_dir, skip_download_check=False, data_augmentation=args.data_aug)
        args.n_classes, args.input_size = 100, 32
    elif args.dataset == 'mnist':
        train_loader, test_loader = get_mnist_dataloaders(args.batch_size, args.test_batch_size, args.num_workers, datadir=dataset_dir)
        args.n_classes, args.input_size = 10, 28
    elif args.dataset == 'ImageNet1k':
        train_loader, test_loader = get_cifar_dataloaders(args.batch_size, args.test_batch_size, 'ImageNet1k', args.num_workers, datadir=dataset_dir, skip_download_check=False, data_augmentation=args.data_aug)
        args.n_classes, args.input_size = 1000, 224
    elif args.dataset == 'ImageNet1k_imagefolder':
        train_loader, test_loader = get_cifar_dataloaders(args.batch_size, args.test_batch_size, 'ImageNet1k_imagefolder', args.num_workers, datadir=dataset_dir, skip_download_check=False, data_augmentation=args.data_aug)
        args.n_classes, args.input_size = 1000, 224
    else:
        raise ValueError('Dataset not supported')
    print(f'Loading dataset done. Time: {time.time()-s0:.2f}s')

    # set model
    _model_params = {  # model parameters
        'num_classes': args.n_classes
    }
    if args.model.startswith('vit'):
        _model_params['image_size'] = args.input_size # manually set image_size for ViT
    _model = get_model(args.model, **_model_params) # get model from models/__init__.py
    _model = _model.to(_device)

    # set optimizer
    _params = _model.parameters()
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(_params, lr=args.lr)
    else:
        optimizer = None
        optimizer = optimizers(_model, args.optimizer, args.lr, args.weight_decay)
        print(f'Using SWD optimizer: {args.optimizer}')

    # set lr scheduler
    if args.lr_scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.5), int(args.epochs*0.75)], gamma=0.1)  # refer to paper: benchopt
    elif args.lr_scheduler == 'exp':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    elif args.lr_scheduler == 'cosine_restart':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=80, T_mult=1)
    elif args.lr_scheduler == 'none' or args.lr_scheduler is None or args.lr_scheduler == 'fixed':
        lr_scheduler = None
    else:
        lr_scheduler = None
        # raise ValueError('lr_scheduler must be one of [cosine, step]')
        lr_scheduler = lr_schedulers(optimizer, args.lr_scheduler)
        print(f'Using SWD lr_scheduler: {args.lr_scheduler}')

    # set criterion
    criterion = nn.CrossEntropyLoss(reduction='mean')

    # set multiclass accuracy metrics
    train_acc = MulticlassAccuracy(num_classes=args.n_classes, device=_device)
    test_acc = MulticlassAccuracy(num_classes=args.n_classes, device=_device)

    # set adversarial attack
    adversary = {
        'train': {},
        'test': {}
    }
    adversary_metrics = {
        'train': {},
        'test': {}
    }
    for adv_type in args.adv_types:
        if adv_type not in ['none', 'pgd_linf', 'pgd_l2']:
            print(f'Adversarial attack type {adv_type} not supported')
            args.adv_types.remove(adv_type)
        
        if adv_type == 'pgd_linf':
            if args.adv_train:
                adversary_train = PGD(_model, eps=args.adv_train_eps, alpha=args.adv_train_alpha, steps=args.adv_train_iters, random_start=args.adv_train_randinit)
                adversary_train.set_device(_device)  # manually set device, since current _model is not on device
                adversary['train'][adv_type] = adversary_train
                adversary_metrics['train'][adv_type] = MulticlassAccuracy(num_classes=args.n_classes, device=_device)
            if args.adv_test:
                adversary_test = PGD(_model, eps=args.adv_test_eps, alpha=args.adv_test_alpha, steps=args.adv_test_iters, random_start=args.adv_test_randinit)
                adversary_test.set_device(_device)  # manually set device, since current _model is not on device
                adversary['test'][adv_type] = adversary_test
                adversary_metrics['test'][adv_type] = MulticlassAccuracy(num_classes=args.n_classes, device=_device)
        elif adv_type == 'pgd_l2':
            if args.adv_train:
                adversary_train = PGDL2(_model, eps=args.adv_train_eps, alpha=args.adv_train_alpha, steps=args.adv_train_iters, random_start=args.adv_train_randinit)
                adversary_train.set_device(_device)  # manually set device, since current _model is not on device
                adversary['train'][adv_type] = adversary_train
                adversary_metrics['train'][adv_type] = MulticlassAccuracy(num_classes=args.n_classes, device=_device)
            if args.adv_test:
                adversary_test = PGDL2(_model, eps=args.adv_test_eps, alpha=args.adv_test_alpha, steps=args.adv_test_iters, random_start=args.adv_test_randinit)
                adversary_test.set_device(_device)  # manually set device, since current _model is not on device
                adversary['test'][adv_type] = adversary_test
                adversary_metrics['test'][adv_type] = MulticlassAccuracy(num_classes=args.n_classes, device=_device)
        else:
            continue

    if not args.distributed or (args.distributed and accelerator.is_main_process):
        # create the root save directory if not exists
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        # make logs dir, and version_n dir within the root save directory if not exists
        if not os.path.exists(os.path.join(args.save_path, 'logs')):
            version = 0
        else:
            # version = len([n for n in os.listdir(os.path.join(args.save_path, 'logs')) if n.startswith('version_') and os.path.isdir(os.path.join(args.save_path, 'logs', n))])
            last_idx = -1
            for n in os.listdir(os.path.join(args.save_path, 'logs')):
                if n.startswith('version_') and os.path.isdir(os.path.join(args.save_path, 'logs', n)):
                    idx = int(n.split('_')[-1])
                    last_idx = max(last_idx, idx)
            version = last_idx + 1
        args.root_save_path = os.path.join(args.save_path, 'logs', f'version_{version}')
        os.makedirs(args.root_save_path)
        os.makedirs(os.path.join(args.root_save_path, 'checkpoints'))

        # record configs as xml
        # with open(os.path.join(args.save_path, 'config.xml'), 'w') as f:
        #     f.write('<config>\n')
        #     for k, v in vars(args).items():
        #         f.write('\t<%s>%s</%s>\n' % (k, v, k))
        #     f.write('</config>\n')
        # record configs as yaml
        with open(os.path.join(args.root_save_path, 'config.yaml'), 'w') as f:
            # _content = {'args': vars(args)}
            _content = {'args': vars(copy.copy(args))}
            yaml.dump(_content, f)
    
    # set shared variables
    SHARED_VARS['args'] = args

    if args.distributed:
        # use accelerate to prepare
        _model, optimizer, train_loader, test_loader, lr_scheduler = accelerator.prepare(
            _model, optimizer, train_loader, test_loader, lr_scheduler
        )
    
    # return args, _device, _model, train_loader, test_loader, optimizer, lr_scheduler, criterion, adversary
    rt_dict = {
        'accelerator': accelerator if args.distributed else None,  # 'accelerator': accelerator,
        'device': _device,
        'model': _model,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler,
        'criterion': criterion,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'adversary': adversary,
        'adversary_metrics': adversary_metrics,
    }
    if not args.distributed or (args.distributed and accelerator.is_main_process):
        print_stats(_model, criterion, optimizer, lr_scheduler)

    return args, rt_dict


def check_early_stop(args):
    current_pth = args.root_save_path
    # check early stop
    if os.path.exists(os.path.join(current_pth, 'killswitch.stop')):
        print('Early stop detected!')
        return True
    return False

    
if __name__ == '__main__':
    basic_setup()
