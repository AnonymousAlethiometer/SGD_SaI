'''
Author: Unknown
Date: 2024-03-01 10:03:11
LastEditTime: 2024-03-01 10:03:13
LastEditors: Unknown
Description: 
FilePath: /Unknown/baseline_train.py
'''
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.zc_proxy import calc_synflow, calc_grad_norm
from utils.optimizer import optimizers
from utils.lr_scheduler import lr_schedulers
from utils.shared_variables import SHARED_VARS
from datasets import get_cifar_dataloaders, get_mnist_dataloaders
from models import get_model
import argparse
import lightning as L
from lightning import seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor
import torchattacks
from torchattacks import PGD, PGDL2

class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0
        self.S = 0  # This will help us to find the variance

    def update(self, x):
        """Update the running mean and variance with a new value."""
        self.n += 1
        old_mean = self.mean
        self.mean += (x - self.mean) / self.n
        self.S += (x - self.mean) * (x - old_mean)
        
    def variance(self):
        """Return the current variance."""
        return self.S / (self.n - 1) if self.n > 1 else 0
    
    def std_dev(self):
        """Return the current standard deviation."""
        return (self.variance())**0.5


class LightningBaseline(L.LightningModule):
    def __init__(self, args, model, criterion, optimizer, lr_scheduler, adversary, cmd_args):
        # super(LightningBaseline, self).__init__()
        super().__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.adversary = adversary
        # self.hparams = hyparams
        # ignore some hyperparameters, and save the rest
        self.save_hyperparameters(ignore=["model", "criterion", "optimizer", "lr_scheduler", "args", "device", "adversary"])
        
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=args.n_classes)
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=args.n_classes)
        if self.args.adv_train:
            # see https://lightning.ai/docs/torchmetrics/latest/pages/overview.html#metrics-and-devices, this proper use can avoid device error
            self.train_acc_adv = nn.ModuleDict({k: torchmetrics.classification.Accuracy(task="multiclass", num_classes=args.n_classes).to(self.device) for k in self.args.adv_types})
                
        if self.args.adv_test:
            self.test_acc_adv = nn.ModuleDict({k: torchmetrics.classification.Accuracy(task="multiclass", num_classes=args.n_classes).to(self.device) for k in self.args.adv_types})

        self.epoch0_snr_mean = 0.
        self.epoch0_snr_max = 0.
        self.epoch0_snr_min = 0.

        self.epoch_snr_mean = 0. 
        self.epoch0_snrs = []
        self.epoch_snrs = []
        self.runing_stats = RunningStats()

    # =================================================================
    # ==================== Life Cycle for training ====================
    # -----------------------------------------------------------------
    def on_after_backward(self):
        # record snr
        self.record_snr(prefix="train_")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # record wd_ratio
        self.log("train_wd/wd_ratio", SHARED_VARS.get('wd_ratio', 1.), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_wd/wd_ratio_adams", SHARED_VARS.get('wd_ratio_adams', 1.), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_wd/wd_ratio_normalized", SHARED_VARS.get('wd_ratio_normalized', 1.), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # self.log("train_wd/wd_ratio_vanilla", SHARED_VARS.get('wd_ratio_vanilla', 1.), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # self.log("train_wd/ema_snr", SHARED_VARS.get('ema_snr', 0.), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # self.log("train_wd/ema_deviation", SHARED_VARS.get('ema_deviation', 0.), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # self.log("train_wd/ema_deviation_v2", SHARED_VARS.get('ema_deviation_v2', 1.), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_wd/sq_grad_norm_mean", SHARED_VARS.get('sq_grad_norm_mean', 0.), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_wd/grad_norm_std_mean", SHARED_VARS.get('grad_norm_std_mean', 0.), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # self.log("train_wd/grad_norm_snr_mean", SHARED_VARS.get('grad_norm_snr_mean', 0.), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # self.log("train_wd/grad_norm_snr_sum", SHARED_VARS.get('grad_norm_snr_sum', 0.), on_step=True, on_epoch=True, prog_bar=False, logger=True)
    
    def on_train_epoch_end(self):
        if self.current_epoch == 0:
 
            self.epoch0_snr_mean = np.mean(self.epoch0_snrs)
            self.epoch0_snr_max = np.max(self.epoch0_snrs)
            self.epoch0_snr_min = np.std(self.epoch0_snrs)
            SHARED_VARS['epoch0_snr_mean'] = self.epoch0_snr_mean
            SHARED_VARS['epoch0_snr_max'] = self.epoch0_snr_max
            SHARED_VARS['epoch0_snr_min'] = self.epoch0_snr_min

        # for x in self.epoch_snrs:
        #     self.runing_stats.update(x)

        # self.epoch_snr_mean = self.runing_stats.mean
        # SHARED_VARS['epoch0_snr_mean'] = self.epoch_snr_mean
        # SHARED_VARS['epoch0_snr_min'] = self.runing_stats.std_dev()
        self.epoch_snrs = []
    # -----------------------------------------------------------------
    # ==================== Life Cycle for training ====================
    # =================================================================

    def record_snr(self, prefix="", is_train=True):
        '''
        record snr and snr related metrics
        invoke by 
            train.on_after_backward 
            val.on_validation_step
        '''
        grads = [p.grad for p in self.model.parameters()]

        syn_score, snrsyn_score = calc_synflow(grads, self.model)
        self.log(f"{prefix}snr/syn_score", syn_score, on_step=True, on_epoch=True, prog_bar=False, logger=True)  # use prefix to distinguish train and val
        self.log(f"{prefix}snr/snrsyn_score", snrsyn_score, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # avg_grad_norm = torch.mean(torch.stack([torch.norm(g) for g in grads]))
        # self.log(f"snr/{prefix}grad_norm", avg_grad_norm, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        grad_norm_score, grad_norm_mean_score, snr_grad_norm_score, snr_grad_norm_mean_score, snr_grad_norm_log1p_score, grad_norm_std_score = calc_grad_norm(grads)
        self.log(f"{prefix}snr/grad_norm", grad_norm_score, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log(f"{prefix}snr/grad_norm_mean_score", grad_norm_mean_score, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log(f"{prefix}snr/snrgn_score", snr_grad_norm_score, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log(f"{prefix}snr/snrgn_mean_score", snr_grad_norm_mean_score, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log(f"{prefix}snr/snrgn_log1p_score", snr_grad_norm_log1p_score, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log(f"{prefix}snr/gn_std", grad_norm_std_score, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        if is_train:
            current_snr = snr_grad_norm_log1p_score  # adjecent to 200, similar curve like mean_snr_score
            SHARED_VARS['step_snr'] = current_snr
            SHARED_VARS['epoch'] = self.current_epoch
            if self.current_epoch == 0:
                # in epoch 0 of training, store each step's snr to calculate mean, max, min
                self.epoch0_snrs.append(current_snr)
                # mean max min calculation will happen at the end of epoch 0 of training
        
            self.epoch_snrs.append(current_snr)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch

        # to calculate robust accuracy
        if self.args.adv_train and (self.current_epoch >= self.args.epochs - self.args.adv_last_nepochs):
            # adversarial training
            for adv_type, adv in self.adversary['train'].items():
                x_adv = adv(x, y)
                x_adv_hat = self.model(x_adv)
                acc_adv = self.train_acc_adv[adv_type](x_adv_hat, y)
                self.log(f"train_acc_adv/{adv_type}", self.train_acc_adv[adv_type], on_step=True, on_epoch=True, prog_bar=False, logger=True)

        x_hat = self.model(x)
        loss = self.criterion(x_hat, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # calculate accuracy
        self.train_acc(x_hat, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.model(x)
        loss = self.criterion(x_hat, y)
        # calculate accuracy
        self.test_acc(x_hat, y)
        self.log("test_acc", self.test_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # manually override and set model to training mode
        torch.set_grad_enabled(True)
        self.model.train()

        x, y = batch

        # to calculate robust accuracy
        if self.args.adv_test and (self.current_epoch >= self.args.epochs - self.args.adv_last_nepochs):
            # adversarial test
            for adv_type, adv in self.adversary['test'].items():
                x_adv = adv(x, y)
                x_adv_hat = self.model(x_adv)
                acc_adv = self.test_acc_adv[adv_type](x_adv_hat, y)
                self.log(f"val_acc_adv/{adv_type}", self.test_acc_adv[adv_type], on_step=True, on_epoch=True, prog_bar=False, logger=True)

        x_hat = self.model(x)
        loss = self.criterion(x_hat, y)
        # calculate accuracy
        acc = self.test_acc(x_hat, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('hp_metric', (1.-acc))

        # manually backprop for recording snr
        loss.backward()
        # record snr
        self.record_snr(prefix="val_", is_train=False)
        
        # resume model to eval mode
        self.model.zero_grad()
        torch.set_grad_enabled(False)
        self.model.eval()

        return loss

    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), lr=1e-3)
        if self.lr_scheduler is None:
            return self.optimizer
        
        # set lr scheduler
        lr_scheduler_config = {
            "scheduler": self.lr_scheduler,
            "interval": "epoch",
            "monitor": "val_loss",
        }

        # return optimizer
        # return self.optimizer
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": lr_scheduler_config,
        }


def train_model(args, model, train_loader, test_loader, optimizer, lr_scheduler, criterion, adversary, device):

    # # fix warning for 3090 tensor core
    # torch.set_float32_matmul_precision('medium' | 'high')

    # change args to dict
    args_dict = vars(args)  # use to save to hparams.yaml, must be serializable

    # init model
    baseline = LightningBaseline(args, model, criterion, optimizer, lr_scheduler=lr_scheduler, adversary=adversary, cmd_args=args_dict)

    # save best model
    checkpoint_best_loss_callback = ModelCheckpoint(
        monitor='val_loss',
        # dirpath=f'{args.save_path}/best_models', # for neat folder structure, don't sepcify dirpath, use default
        filename='best_loss-{epoch:02d}-{val_loss:.3f}-{val_acc:.4f}-{hp_metric:.4f}',
        save_top_k=1,
        mode='min',
    )
    checkpoint_best_acc_callback = ModelCheckpoint(
        monitor='val_acc',
        # dirpath=f'{args.save_path}/best_models', # for neat folder structure, don't sepcify dirpath, use default
        filename='best_acc-{epoch:02d}-{val_loss:.3f}-{val_acc:.4f}-{hp_metric:.4f}',
        save_top_k=1,
        mode='max',
    )
    checkpoint_last_callback = ModelCheckpoint(
        monitor=None,  # monitor is set to None, so it will save model only at the end of training
        # dirpath=f'{args.save_path}/best_models', # for neat folder structure, don't sepcify dirpath, use default
        filename='last-{epoch:02d}-{val_loss:.3f}-{val_acc:.4f}-{hp_metric:.4f}',
        every_n_epochs=args.epochs,
    )
    ckpt_callbacks = [checkpoint_best_loss_callback, checkpoint_best_acc_callback, checkpoint_last_callback]
    
    # save learning rate
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # train the model
    if args.early_stop:
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10, verbose=False, mode="min")
        trainer = L.Trainer(default_root_dir=args.save_path, callbacks=[early_stop_callback, *ckpt_callbacks, lr_monitor], max_epochs=args.epochs)
    else:
        # trainer = L.Trainer(limit_train_batches=200, max_epochs=3, default_root_dir=args.save_path)
        trainer = L.Trainer(default_root_dir=args.save_path, callbacks=[*ckpt_callbacks, lr_monitor], max_epochs=args.epochs)
    trainer.fit(model=baseline, train_dataloaders=train_loader, val_dataloaders=test_loader, )


def basic_setup():

    parser = argparse.ArgumentParser(description='PyTorch Baseline Training')
    parser.add_argument('--model', default='resnet50', type=str, help='model name')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
    parser.add_argument('--data_dir', default='./', type=str, help='dataset root directory')
    parser.add_argument('--data_aug', action='store_true', help='whether to use data augmentation')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--batch_size', default=128, type=int, help='train batch size')
    parser.add_argument('--test_batch_size', default=128, type=int, help='test batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='dataloader num_workers')
    parser.add_argument('--epochs', default=200, type=int, help='train epochs')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_scheduler', default='none', type=str, help='learning rate scheduler (cosine, cosine_restart, step, exp, none, fixed)')
    parser.add_argument('--optimizer', default='sgd', type=str, help='optimizer (sgd, adam, adamw, rmsprop)')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='SGD/Adam/AdamW weight decay')
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint path')
    parser.add_argument('--save_path', default='checkpoints', type=str, help='save checkpoint path')
    parser.add_argument('--early_stop', action='store_true', help='whether to use early stop')
    parser.add_argument('--gpu', default='0', type=str, help='gpu id')
    parser.add_argument('--wd_ratio_monitor', action='store_true', help='whether to monitor the weight decay rate')
    # adversarial attack
    parser.add_argument('--adv_types', nargs='+', default=['none'], type=str, help='adversarial attack types, e.g., pgd_linf, pgd_l2')
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
    parser.add_argument('--adv_last_nepochs', default=5, type=int, help='last n epochs to attack')


    args = parser.parse_args()

    # set gpu id
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    seed_everything(args.seed, workers=True)

    # set device
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set dataloader
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    if args.dataset == 'cifar10':
        train_loader, test_loader = get_cifar_dataloaders(args.batch_size, args.test_batch_size, 'cifar10', args.num_workers, datadir=dataset_dir, skip_download_check=False, data_augmentation=args.data_aug)
        args.n_classes = 10
    elif args.dataset == 'cifar100':
        train_loader, test_loader = get_cifar_dataloaders(args.batch_size, args.test_batch_size, 'cifar100', args.num_workers, datadir=dataset_dir, skip_download_check=False, data_augmentation=args.data_aug)
        args.n_classes = 100
    elif args.dataset == 'mnist':
        train_loader, test_loader = get_mnist_dataloaders(args.batch_size, args.test_batch_size, args.num_workers, datadir=dataset_dir)
        args.n_classes = 10

    # set model
    _model = get_model(args.model, num_classes=args.n_classes)


    SHARED_VARS['args'] = args

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
    criterion = nn.CrossEntropyLoss()

    # set adversarial attack
    adversary = {
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
            if args.adv_test:
                adversary_test = PGD(_model, eps=args.adv_test_eps, alpha=args.adv_test_alpha, steps=args.adv_test_iters, random_start=args.adv_test_randinit)
                adversary_test.set_device(_device)  # manually set device, since current _model is not on device
                adversary['test'][adv_type] = adversary_test
        elif adv_type == 'pgd_l2':
            if args.adv_train:
                adversary_train = PGDL2(_model, eps=args.adv_train_eps, alpha=args.adv_train_alpha, steps=args.adv_train_iters, random_start=args.adv_train_randinit)
                adversary_train.set_device(_device)  # manually set device, since current _model is not on device
                adversary['train'][adv_type] = adversary_train
            if args.adv_test:
                adversary_test = PGDL2(_model, eps=args.adv_test_eps, alpha=args.adv_test_alpha, steps=args.adv_test_iters, random_start=args.adv_test_randinit)
                adversary_test.set_device(_device)  # manually set device, since current _model is not on device
                adversary['test'][adv_type] = adversary_test
        else:
            continue

    # save args to checkpoints dir as config.xml
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    with open(os.path.join(args.save_path, 'config.xml'), 'w') as f:
        f.write('<config>\n')
        for k, v in vars(args).items():
            f.write('\t<%s>%s</%s>\n' % (k, v, k))
        f.write('</config>\n')

    return args, _device, _model, train_loader, test_loader, optimizer, lr_scheduler, criterion, adversary


if __name__ == '__main__':

    args, device, model, train_loader, test_loader, optimizer, lr_scheduler, criterion, adversary = basic_setup()
    print(args)

    # train model
    train_model(args, model, train_loader, test_loader, optimizer, lr_scheduler, criterion, adversary, device)

    print('='*20)
    print('done')
    print('-'*15)
    print('- config:')
    for k, v in vars(args).items():
        print(f'\t- {k}: {v}')
    print('='*20)

