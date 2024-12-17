'''
Author: Unknown
Date: 2024-03-01 10:05:06
LastEditTime: 2024-05-17 20:52:02
LastEditors: Unknown
Description: 
FilePath: /Unknown/vanilla_baseline_train.py
'''
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from utils.train_utils import basic_setup, check_early_stop
from utils.misc import AverageMeter, AverageMeterManager
from utils.shared_variables import SHARED_VARS
from utils.zc_proxy import calc_synflow, calc_grad_norm
from utils.optimizer import record_optimizer_info
import time

has_tqdm = False
try:
    from tqdm import tqdm
    has_tqdm = True
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x

has_wandb = False
WANDB_PROJECT_NAME = 'Unknown-GS'
try:
    import wandb
    has_wandb = True
except ImportError:
    pass


class Trainer():
    def __init__(self, args, *argv, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                raise ValueError(f'key: {k} is None')

        self.args = args
        self.device = kwargs.get('device', None)
        self.model = kwargs.get('model', None)
        self.train_loader = kwargs.get('train_loader', None)
        self.test_loader = kwargs.get('test_loader', None)
        self.optimizer = kwargs.get('optimizer', None)
        self.lr_scheduler = kwargs.get('lr_scheduler', None)
        self.criterion = kwargs.get('criterion', None)
        self.adversary = kwargs.get('adversary', None)
        self.adversary_metrics = kwargs.get('adversary_metrics', None)
        self.train_acc = kwargs.get('train_acc', None)
        self.test_acc = kwargs.get('test_acc', None)

        # tensorboard
        self.writter = SummaryWriter(log_dir=args.root_save_path)

        # metrics
        self.best_acc = -1. # epoch best acc
        self.best_acc_epochid = -1. # best acc epoch id
        self.best_loss = torch.inf # epoch best loss
        self.best_loss_epochid = -1 # best loss epoch id
        # self.train_loss = AverageMeter()
        # self.test_loss = AverageMeter()
        self.metric_mgr = AverageMeterManager()


        # states
        self.current_epoch = -1
        self.current_step = -1

        self.epoch0_snr_mean = 0.
        self.epoch0_snr_max = 0.
        self.epoch0_snr_min = 0.

        self.epoch_snr_mean = 0. 
        self.epoch0_snrs = []
        self.epoch_snrs = []

        # save hyperparameters to tensorboard
        # args_dict = vars(args) # potential bug: args is a Namespace object, which is mutable
        args_dict = copy.deepcopy(args.__dict__)  # avoid changing the original args, which may cause bugs for adv_train and adv_test
        # replace v types:
        for k, v in args_dict.items():
            if type(v) in [list, dict, set, tuple]:
                args_dict[k] = str(v)
        self.writter.add_hparams(args_dict, {}, run_name='.')

        if has_wandb:
            run = wandb.init(project=WANDB_PROJECT_NAME, config=args_dict)
    
    # ===============================================
    # ==================== train ====================
    # ===============================================

    def on_training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        # to calculate robust accuracy
        if self.args.adv_train and self.is_adv_last_nepochs:
            # adversarial training
            for adv_type, adv in self.adversary['train'].items():
                x_adv = adv(x, y)
                x_adv_hat = self.model(x_adv)
                # calculate robust accuracy for each step
                step_robust_acc = (torch.argmax(x_adv_hat, dim=1) == y).sum().item() / y.size(0)
                # update robust accuracy
                self.adversary_metrics['train'][adv_type].update(x_adv_hat, y)
                # log to tensorboard
                self.writter.add_scalar(f'train/robust_acc_{adv_type}_step', step_robust_acc, self.current_step)

        x_hat = self.model(x)
        loss = self.criterion(x_hat, y)
        # calculate step accuracy
        # step_acc = (torch.argmax(x_hat, dim=1) == y).sum().item() / y.size(0)
        step_acc = (torch.argmax(x_hat, dim=1) == y).sum() / y.size(0)
        
        # record of epoch metrics for tensorboard
        # self.train_loss.update(loss)
        self.metric_mgr.update('loss', loss.item(), is_train=True, n=y.size(0))
        self.train_acc.update(x_hat, y)
        # self.metric_mgr.update('loss_new', loss.item(), is_train=True, n=y.size(0))
        # self.metric_mgr.update('acc_new', step_acc, is_train=True, n=y.size(0))
        
        # log to tensorboard
        self.writter.add_scalar('train/loss_step', loss, self.current_step)
        self.writter.add_scalar('train/acc_step', step_acc, self.current_step)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # record wd_ratio
        self.metric_mgr.update('wd/wd_ratio', SHARED_VARS.get('wd_ratio', 1.), is_train=True)
        self.metric_mgr.update('wd/wd_ratio_adams', SHARED_VARS.get('wd_ratio_adams', 1.), is_train=True)
        self.metric_mgr.update('wd/wd_ratio_normalized', SHARED_VARS.get('wd_ratio_normalized', 1.), is_train=True)
        # self.log("train_wd/wd_ratio", SHARED_VARS.get('wd_ratio', 1.), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # self.log("train_wd/wd_ratio_adams", SHARED_VARS.get('wd_ratio_adams', 1.), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # self.log("train_wd/wd_ratio_normalized", SHARED_VARS.get('wd_ratio_normalized', 1.), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
        # self.log("train_wd/wd_ratio_vanilla", SHARED_VARS.get('wd_ratio_vanilla', 1.), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # self.log("train_wd/ema_snr", SHARED_VARS.get('ema_snr', 0.), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # self.log("train_wd/ema_deviation", SHARED_VARS.get('ema_deviation', 0.), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # self.log("train_wd/ema_deviation_v2", SHARED_VARS.get('ema_deviation_v2', 1.), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
        self.metric_mgr.update('wd/sq_grad_norm_mean', SHARED_VARS.get('sq_grad_norm_mean', 0.), is_train=True)
        self.metric_mgr.update('wd/grad_norm_std_mean', SHARED_VARS.get('grad_norm_std_mean', 0.), is_train=True)
        # self.log("train_wd/sq_grad_norm_mean", SHARED_VARS.get('sq_grad_norm_mean', 0.), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # self.log("train_wd/grad_norm_std_mean", SHARED_VARS.get('grad_norm_std_mean', 0.), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # self.log("train_wd/grad_norm_snr_mean", SHARED_VARS.get('grad_norm_snr_mean', 0.), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # self.log("train_wd/grad_norm_snr_sum", SHARED_VARS.get('grad_norm_snr_sum', 0.), on_step=True, on_epoch=True, prog_bar=False, logger=True)
    
    def on_optimizer_warmup(self):
        # warmup for AdamSNR
        self.model.train()
        tqdm_iter = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f'[Warmup] Epoch -1', leave=False)
        for batch_idx, batch in tqdm_iter:
            # self.current_step += 1
            self.optimizer.zero_grad()

            # [lifecycle] training step
            loss = self.on_training_step(batch, batch_idx)
            loss.backward()

            # # record snr
            # self.record_snr(is_train=True)

            self.optimizer.warm_up()
            
            # # log to tensorboard
            # self.writter.add_scalar('epoch', self.current_epoch, self.current_step)

            # # [lifecycle] triggerd on batch end
            # self.on_train_batch_end(loss, batch, batch_idx)

            # update tqdm
            if has_tqdm:
                tqdm_iter.set_postfix({'loss': loss.item()})

        # # trigger testing
        # self.test()
        # self.lr_scheduler.step()

    def on_train_epoch(self):
        loss = torch.inf
        tqdm_iter = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f'[Train] Epoch {self.current_epoch}/{self.args.epochs}', leave=False)
        for batch_idx, batch in tqdm_iter:
            self.current_step += 1  # self.current_step is 0 for the first step of the first epoch
            self.optimizer.zero_grad()

            # adjust learning rate for warmup manually
            if self.current_step < self.args.warmup_steps:
                lr = self.args.lr * (self.current_step+1) / self.args.warmup_steps
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            # log lr
            # print(f'current_step: {self.current_step}/{self.args.warmup_steps}, effective_lr: {self.optimizer.param_groups[0]["lr"]}')
            # with open(os.path.join(self.args.root_save_path, 'lr_log.csv'), 'a') as f:
            #     f.write(f'{self.current_step},{self.optimizer.param_groups[0]["lr"]}\n')
            # continue

            # [lifecycle] training step
            loss = self.on_training_step(batch, batch_idx)
            loss.backward()

            # record snr
            self.record_snr(is_train=True)

            if not hasattr(self.optimizer, 'has_warmup') and hasattr(self.optimizer, 'warmup_step'):
                self.optimizer.warmup_step()
                self.optimizer.has_warmup = True

            self.optimizer.step()
            # record_optimizer_info(self.optimizer, self.writter, self.current_step, self.current_epoch, extra_save_pth=self.args.root_save_path)  #TODO: save gradient info; 

            # log to tensorboard
            self.writter.add_scalar('epoch', self.current_epoch, self.current_step)

            # [lifecycle] triggerd on batch end
            self.on_train_batch_end(loss, batch, batch_idx)

            # update tqdm
            if has_tqdm:
                tqdm_iter.set_postfix({'loss': loss.item()})
            
            # early stop
            if check_early_stop(self.args):
                break


        # trigger testing
        self.test()
        self.lr_scheduler.step() if self.current_step >= self.args.warmup_steps else None  # skip lr_scheduler.step() during warmup


    def train(self):
        self.training_progress = {
            f'{self.args.optimizer}_test_err': [],
            # f'{self.args.optimizer}_test_err_new': [],
            f'{self.args.optimizer}_train_err': [],
            # f'{self.args.optimizer}_train_err_new': [],
            f'{self.args.optimizer}_train_loss': [],
            # f'{self.args.optimizer}_train_loss_new': [],
        }
        self.adv_progress = { # for adversarial training and testing acc record
            f'{self.args.optimizer}_epoch': [],
            **{f'{self.args.optimizer}_{adv_type}_train_acc': [] for adv_type in self.args.adv_types if self.args.adv_train},
            **{f'{self.args.optimizer}_{adv_type}_test_acc': [] for adv_type in self.args.adv_types if self.args.adv_test},
        }


        # record start time
        readable_start_time = time.strftime("%Y-%m-%d__%H_%M_%S", time.localtime())
        with open(os.path.join(self.args.root_save_path, f'00_start_time__{readable_start_time}.txt'), 'w') as f:
            f.write(readable_start_time)

        # if type(self.optimizer) in [SGD_boost]:
        #     # let us do optimizer warmup
        #     self.on_optimizer_warmup()


        for epoch in range(self.args.epochs):
            # set back to train mode
            torch.set_grad_enabled(True)
            self.model.train()
            self.current_epoch = epoch
            # loss = torch.inf
            self.is_adv_last_nepochs = (self.current_epoch >= self.args.epochs - self.args.adv_last_nepochs)
            
            # train this epoch
            self.on_train_epoch()
            
            # early stop
            if check_early_stop(self.args):
                print('*'*20)
                print('* Early stop detected, mannually stop training!')
                print('*'*10)
                print('* Current epoch:', self.current_epoch)
                print('* Current step:', self.current_step)
                print('* Current best acc:', self.best_acc)
                print('*'*20)
                # record end time
                readable_skip_time = time.strftime("%Y-%m-%d__%H_%M_%S", time.localtime())
                with open(os.path.join(self.args.root_save_path, f'02_earlystop_time__{readable_skip_time}.txt'), 'w') as f:
                    f.write(readable_skip_time)
                break

            # log to tensorboard
            # self.writter.add_scalar('train/loss_epoch', self.train_loss.compute(), self.current_step)
            self.writter.add_scalar('train/loss_epoch', self.metric_mgr.compute('loss', is_train=True), self.current_step)
            self.writter.add_scalar('train/acc_epoch', self.train_acc.compute(), self.current_step)
            if self.args.adv_train and self.is_adv_last_nepochs:
                if self.current_epoch not in self.adv_progress[f'{self.args.optimizer}_epoch']:
                    self.adv_progress[f'{self.args.optimizer}_epoch'].append(self.current_epoch)
                for adv_type, adv in self.adversary['train'].items():
                    self.writter.add_scalar(f'train/robust_acc_{adv_type}_epoch', self.adversary_metrics['train'][adv_type].compute(), self.current_step)
                    self.adv_progress[f'{self.args.optimizer}_{adv_type}_train_acc'].append(self.adversary_metrics['train'][adv_type].compute().detach().cpu().item())
            for metric_name in ['wd/wd_ratio', 'wd/wd_ratio_adams', 'wd/wd_ratio_normalized', 'wd/sq_grad_norm_mean', 'wd/grad_norm_std_mean']:
                self.writter.add_scalar(f'train_{metric_name}_epoch', self.metric_mgr.compute(metric_name, is_train=True), self.current_step)
            self.training_progress[f'{self.args.optimizer}_train_err'].append(1.-self.train_acc.compute().detach().cpu().item())
            self.training_progress[f'{self.args.optimizer}_train_loss'].append(self.metric_mgr.compute('loss', is_train=True))
            # self.training_progress[f'{self.args.optimizer}_train_err_new'].append(1.-self.metric_mgr.compute('acc_new', is_train=True).detach().cpu().item())
            # self.training_progress[f'{self.args.optimizer}_train_loss_new'].append(self.metric_mgr.compute('loss_new', is_train=True))
            if has_wandb:
                wandb.log({
                    'train/loss_epoch': self.metric_mgr.compute('loss', is_train=True),
                    'train/acc_epoch': self.train_acc.compute(),
                    'epoch': self.current_epoch,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                }, step=self.current_epoch, commit=False)

            # reset metrics
            # self.train_loss.reset()
            self.metric_mgr.reset(is_train=True)
            self.train_acc.reset()
            if self.args.adv_train and self.is_adv_last_nepochs:
                for adv_type, adv in self.adversary['train'].items():
                    self.adversary_metrics['train'][adv_type].reset()
            
            # reset snr anchor results.
            if self.current_epoch == 0:
    
                self.epoch0_snr_mean = np.mean(self.epoch0_snrs)
                self.epoch0_snr_max = np.max(self.epoch0_snrs)
                self.epoch0_snr_min = np.std(self.epoch0_snrs)
                SHARED_VARS['epoch0_snr_mean'] = self.epoch0_snr_mean
                SHARED_VARS['epoch0_snr_max'] = self.epoch0_snr_max
                SHARED_VARS['epoch0_snr_min'] = self.epoch0_snr_min

            self.epoch_snrs = []


            # save training progress
            df = pd.DataFrame(self.training_progress)
            df.to_csv(os.path.join(self.args.root_save_path, 'training_progress.csv'), index=False)

            # save adv progress
            df = pd.DataFrame(self.adv_progress)
            df.to_csv(os.path.join(self.args.root_save_path, 'adv_progress.csv'), index=False)

        # clean up
        self.writter.flush()
        self.writter.close()

        # record end time
        readable_end_time = time.strftime("%Y-%m-%d__%H_%M_%S", time.localtime())
        with open(os.path.join(self.args.root_save_path, f'01_end_time__{readable_end_time}.txt'), 'w') as f:
            f.write(readable_end_time)
    
    
    def on_testing_step(self, batch, batch_idx):
        # manually override and set model to training mode
        torch.set_grad_enabled(True)
        # self.model.train()
        self.model.eval()  # set to eval mode, so that batchnorm and dropout will be disabled

        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        if self.args.adv_test and self.is_adv_last_nepochs:
            # adversarial testing
            for adv_type, adv in self.adversary['test'].items():
                x_adv = adv(x, y)
                x_adv_hat = self.model(x_adv)
                
                # calculate robust accuracy for each step
                step_robust_acc = (torch.argmax(x_adv_hat, dim=1) == y).sum().item() / y.size(0)
                # update robust accuracy
                self.adversary_metrics['test'][adv_type].update(x_adv_hat, y)
                # log to tensorboard
                self.writter.add_scalar(f'test/robust_acc_{adv_type}_step', step_robust_acc, self.current_step)
        
        x_hat = self.model(x)
        loss = self.criterion(x_hat, y)
        # manually backprop for recording snr
        loss.backward()

        # record snr
        self.record_snr(is_train=False)

        # calculate accuracy
        # step_acc = (torch.argmax(x_hat, dim=1) == y).sum().item() / y.size(0)
        step_acc = (torch.argmax(x_hat, dim=1) == y).sum() / y.size(0)

        # record of epoch metrics for tensorboard
        # self.test_loss.update(loss)
        self.metric_mgr.update('loss', loss.item(), is_train=False, n=y.size(0))
        self.test_acc.update(x_hat, y)
        # self.metric_mgr.update('loss_new', loss.item(), is_train=False, n=y.size(0))
        # self.metric_mgr.update('acc_new', step_acc, is_train=False, n=y.size(0))

        # log to tensorboard
        self.writter.add_scalar('test/loss_step', loss, self.current_step)
        self.writter.add_scalar('test/acc_step', step_acc, self.current_step)

        # resume model to eval mode
        self.model.zero_grad()
        torch.set_grad_enabled(False)
        # self.model.eval()
        return loss


    def test(self):
        self.model.eval()
        tqdm_iter = tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc=f'[Test] Epoch {self.current_epoch}/{self.args.epochs}', leave=False)
        for batch_idx, batch in tqdm_iter:
            # [lifecycle] testing step
            loss = self.on_testing_step(batch, batch_idx)

            # update tqdm
            if has_tqdm:
                tqdm_iter.set_postfix({'loss': loss.item()})

        # log to tensorboard
        # self.writter.add_scalar('test/loss_epoch', self.test_loss.compute(), self.current_step)
        self.writter.add_scalar('test/loss_epoch', self.metric_mgr.compute('loss', is_train=False), self.current_step)
        self.writter.add_scalar('test/acc_epoch', self.test_acc.compute(), self.current_step)
        self.writter.add_scalar('hp_metric', (1.-self.test_acc.compute()), self.current_step)
        if self.args.adv_test and self.is_adv_last_nepochs:
            if self.current_epoch not in self.adv_progress[f'{self.args.optimizer}_epoch']:
                self.adv_progress[f'{self.args.optimizer}_epoch'].append(self.current_epoch)
            for adv_type, adv in self.adversary['test'].items():
                self.writter.add_scalar(f'test/robust_acc_{adv_type}_epoch', self.adversary_metrics['test'][adv_type].compute(), self.current_step)
                self.adv_progress[f'{self.args.optimizer}_{adv_type}_test_acc'].append(self.adversary_metrics['test'][adv_type].compute().detach().cpu().item())
        self.training_progress[f'{self.args.optimizer}_test_err'].append(1.-self.test_acc.compute().detach().cpu().item())
        # self.training_progress[f'{self.args.optimizer}_test_loss'].append(self.metric_mgr.compute('loss', is_train=False))
        # self.training_progress[f'{self.args.optimizer}_test_err_new'].append(1.-self.metric_mgr.compute('acc_new', is_train=False).detach().cpu().item())
        if has_wandb:
            wandb.log({
                'test/loss_epoch': self.metric_mgr.compute('loss', is_train=False),
                'test/acc_epoch': self.test_acc.compute(),
                'eval_top1': self.test_acc.compute(),
            }, step=self.current_epoch)

        # save best model
        self.save_best_model()

        # save last model
        if self.current_epoch == self.args.epochs - 1:
            self.save_last_model()

        # reset metrics
        # self.test_loss.reset()
        self.metric_mgr.reset(is_train=False)
        self.test_acc.reset()
        if self.args.adv_test and self.is_adv_last_nepochs:
            for adv_type, adv in self.adversary['test'].items():
                self.adversary_metrics['test'][adv_type].reset()
    # ===============================================
    # ==================== misc ====================
    # ===============================================

    def record_snr(self, is_train=True):
        '''
        record snr and snr related metrics
        invoke by 
            train.on_after_backward 
            val.on_validation_step
        '''
        grads = [p.grad for p in self.model.parameters()]

        syn_score, snrsyn_score = calc_synflow(grads, self.model)
        self.metric_mgr.update('snr/syn_score', syn_score, is_train=is_train)
        self.metric_mgr.update('snr/snrsyn_score', snrsyn_score, is_train=is_train)
        # self.log(f"{prefix}snr/syn_score", syn_score, on_step=True, on_epoch=True, prog_bar=False, logger=True)  # use prefix to distinguish train and val
        # self.log(f"{prefix}snr/snrsyn_score", snrsyn_score, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # avg_grad_norm = torch.mean(torch.stack([torch.norm(g) for g in grads]))
        # self.log(f"snr/{prefix}grad_norm", avg_grad_norm, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        grad_norm_score, grad_norm_mean_score, snr_grad_norm_score, snr_grad_norm_mean_score, snr_grad_norm_log1p_score, grad_norm_std_score = calc_grad_norm(grads)
        self.metric_mgr.update('snr/grad_norm', grad_norm_score, is_train=is_train)
        self.metric_mgr.update('snr/grad_norm_mean_score', grad_norm_mean_score, is_train=is_train)
        self.metric_mgr.update('snr/snrgn_score', snr_grad_norm_score, is_train=is_train)
        self.metric_mgr.update('snr/snrgn_mean_score', snr_grad_norm_mean_score, is_train=is_train)
        self.metric_mgr.update('snr/snrgn_log1p_score', snr_grad_norm_log1p_score, is_train=is_train)
        self.metric_mgr.update('snr/gn_std', grad_norm_std_score, is_train=is_train)
        # self.log(f"{prefix}snr/grad_norm", grad_norm_score, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # self.log(f"{prefix}snr/grad_norm_mean_score", grad_norm_mean_score, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # self.log(f"{prefix}snr/snrgn_score", snr_grad_norm_score, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # self.log(f"{prefix}snr/snrgn_mean_score", snr_grad_norm_mean_score, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # self.log(f"{prefix}snr/snrgn_log1p_score", snr_grad_norm_log1p_score, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # self.log(f"{prefix}snr/gn_std", grad_norm_std_score, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        if is_train:
            current_snr = snr_grad_norm_log1p_score  # adjecent to 200, similar curve like mean_snr_score
            SHARED_VARS['step_snr'] = current_snr
            SHARED_VARS['epoch'] = self.current_epoch
            if self.current_epoch == 0:
                # in epoch 0 of training, store each step's snr to calculate mean, max, min
                self.epoch0_snrs.append(current_snr)
                # mean max min calculation will happen at the end of epoch 0 of training
            self.epoch_snrs.append(current_snr)


    def save_best_model(self):
        if self.test_acc.compute() > self.best_acc:
            self.best_acc = self.test_acc.compute()
            self.best_acc_epochid = self.current_epoch
            avg_loss = self.metric_mgr.compute('loss', is_train=False)
            # delete old model
            for fname in os.listdir(os.path.join(self.args.root_save_path, 'checkpoints')):
                if fname.startswith('best_acc'):
                    os.remove(os.path.join(self.args.root_save_path, 'checkpoints', fname))
            # save model
            # torch.save(self.model.state_dict(), os.path.join(self.args.root_save_path, 'checkpoints', f'best_acc-epoch={self.best_acc_epochid}-step={self.current_step}-val_acc={self.best_acc:.4f}-val_loss={avg_loss:.4f}-hp_metric={1.-self.best_acc:.4f}.pth'))
            torch.save({
                'epoch': self.current_epoch,
                'step': self.current_step,
                'val_acc': self.best_acc,
                'val_loss': avg_loss,
                'hp_metric': (1.-self.best_acc),
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, os.path.join(self.args.root_save_path, 'checkpoints', f'best_acc-epoch={self.best_acc_epochid}-step={self.current_step}-val_acc={self.best_acc:.4f}-val_loss={avg_loss:.4f}-hp_metric={1.-self.best_acc:.4f}.pth'))
            # save best results
            with open(os.path.join(self.args.root_save_path, 'best_results.csv'), 'w') as f:
                f.write(f'best_acc_epochid,best_loss_epochid,best_val_acc,best_val_loss,hp_metric\n')
                f.write(f'{self.best_acc_epochid},{self.best_loss_epochid},{self.best_acc:.4f},{self.best_loss:.4f},{1.-self.best_acc:.4f}\n')
                f.write(f'{self.best_acc_epochid},{self.best_loss_epochid},{self.best_acc},{self.best_loss},{1.-self.best_acc}\n')
        if self.metric_mgr.compute('loss', is_train=False) < self.best_loss:
            self.best_loss = self.metric_mgr.compute('loss', is_train=False)
            self.best_loss_epochid = self.current_epoch
            avg_acc = self.test_acc.compute()
            # delete old model
            for fname in os.listdir(os.path.join(self.args.root_save_path, 'checkpoints')):
                if fname.startswith('best_loss'):
                    os.remove(os.path.join(self.args.root_save_path, 'checkpoints', fname))
            # save model
            # torch.save(self.model.state_dict(), os.path.join(self.args.root_save_path, 'checkpoints', f'best_loss-epoch={self.best_loss_epochid}-step={self.current_step}-val_acc={avg_acc:.4f}-val_loss={self.best_loss:.4f}-hp_metric={1.-self.best_acc:.4f}.pth'))
            torch.save({
                'epoch': self.current_epoch,
                'step': self.current_step,
                'val_acc': avg_acc,
                'val_loss': self.best_loss,
                'hp_metric': (1.-avg_acc),
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, os.path.join(self.args.root_save_path, 'checkpoints', f'best_loss-epoch={self.best_loss_epochid}-step={self.current_step}-val_acc={avg_acc:.4f}-val_loss={self.best_loss:.4f}-hp_metric={1.-self.best_acc:.4f}.pth'))
            # save best results
            with open(os.path.join(self.args.root_save_path, 'best_results.csv'), 'w') as f:
                f.write(f'best_acc_epochid,best_loss_epochid,best_val_acc,best_val_loss,hp_metric\n')
                f.write(f'{self.best_acc_epochid},{self.best_loss_epochid},{self.best_acc:.4f},{self.best_loss:.4f},{1.-self.best_acc:.4f}\n')
                f.write(f'{self.best_acc_epochid},{self.best_loss_epochid},{self.best_acc},{self.best_loss},{1.-self.best_acc}\n')

    def save_last_model(self):
        # save model
        # torch.save(self.model.state_dict(), os.path.join(self.args.root_save_path, 'checkpoints', f'last-epoch={self.current_epoch}-step={self.current_step}.pth'))
        avg_acc = self.test_acc.compute()
        avg_loss = self.metric_mgr.compute('loss', is_train=False)
        torch.save({
            'epoch': self.current_epoch,
            'step': self.current_step,
            'val_acc': avg_acc,
            'val_loss': avg_loss,
            'hp_metric': (1.-avg_acc),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(self.args.root_save_path, 'checkpoints', f'last-epoch={self.current_epoch}-step={self.current_step}-val_acc={avg_acc:.4f}-val_loss={avg_loss:.4f}-hp_metric={1.-avg_acc:.4f}.pth'))


if __name__ == '__main__':
    print('='*20)
    print('===> start')
    print('='*20)

    args, rt_dict = basic_setup()
    print('='*20)
    print(args)
    print('='*20)
    
    if has_wandb:
        wandb.login()

    # train model
    trainer = Trainer(args, **rt_dict)
    trainer.train()

    if has_wandb:
        wandb.finish()

    print('='*20)
    print('done')
    print('-'*15)
    print('- config:')
    for k, v in vars(args).items():
        print(f'\t- {k}: {v}')
    print('='*20)

