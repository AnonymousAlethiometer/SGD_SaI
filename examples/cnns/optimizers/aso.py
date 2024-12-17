'''
Author: Unknown
Date: 2024-04-07 11:02:01
LastEditTime: 2024-05-24 12:03:25
LastEditors: Unknown
Description: calculate sample-wise SNR and update term for AdamXV2
FilePath: /Unknown/optimizers/aso.py
'''
import math
import torch
from torch.optim.optimizer import Optimizer
import numpy as np


def sigmoid(x):
    if x < 0:
        x *= 2
    """Compute the sigmoid of x."""
    return 2 / (1 + np.exp(-x))

class AutoSampleOptimizer(Optimizer):
    r"""Implements Adam with stable weight decay (AdamS) algorithm.
    It has be proposed in 
    `Stable Weight Decay Regularization`__.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-4)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-4, amsgrad=False, variant='v1'):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, variant=variant)
        super(AutoSampleOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AutoSampleOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        param_size = 0
        # snr
        # grad_norm_snr_sum = 0.
        S_t_sum = 0.
        exp_avg_sq_hat_sum = 0.
        # param_count = 0

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_size += p.numel() # Count the number of elements in the parameter

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamSNR does not support sparse gradients')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                    state['norm'] = 0
                    state['norm_snr'] = 0
                    state['norm_std'] = 0

                    state['S_t'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['term_update'] = 0


                beta1, beta2 = group['betas']
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1
                bias_correction2 = 1 - beta2 ** state['step']
                bias_correction1 = 1 - beta1 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # if amsgrad:
                #     max_exp_avg_sq = state['max_exp_avg_sq']
                #     # Maintains the maximum of all 2nd moment running avg. till now
                #     torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                #     # Use the max. for normalizing running avg. of gradient
                #     exp_avg_sq_hat = max_exp_avg_sq / bias_correction2
                # else:
                #     exp_avg_sq_hat = exp_avg_sq / bias_correction2
                
                term_m_hat = exp_avg / bias_correction1
                term_v_hat = exp_avg_sq_hat


                # calculate layer-wise grad_norm with std
                sigma = torch.std(grad)
                sigma = torch.tensor(0.) if torch.isnan(sigma) else sigma

                grad_norm = grad.norm()
                grad_norm_snr = (grad_norm / sigma) if sigma != 0 else grad_norm
                
                state['norm'] = grad_norm.item()
                state['norm_snr'] = grad_norm_snr.item()
                state['norm_std'] = sigma.item()

                if group['variant'] == 'v1': #not working
                    term_S_t = (term_m_hat * grad_norm_snr) / (term_v_hat.sqrt() + group['eps'])
                    # state['S_t'] = term_S_t
                elif group['variant'] == 'v2':
                    term_S_t = (term_m_hat * grad_norm_snr)
                elif group['variant'] == 'v3': #not working
                    term_S_t = (term_m_hat * (sigma + group['eps'])) / (term_v_hat.sqrt() + group['eps'])
                elif group['variant'] in ['v4', 'v5', 'v6', 'v7']:
                    term_S_t = (term_m_hat * grad_norm_snr)

                state['S_t'] = term_S_t  # for Algorithm 3
                S_t_sum += term_S_t.abs().sum()  # for Algorithm 2
                term_update = group['lr'] * term_S_t  # calculate true update term
                state['term_update'] = term_update
                exp_avg_sq_hat_sum += exp_avg_sq_hat.sum()  # for Algorithm 4: replace Algo2 AdamW-like weight decay with AdamS-like WD

        S_t_mean = S_t_sum / param_size  # for Algorithm 2
        # Calculate the sqrt of the mean of all elements in exp_avg_sq_hat  
        exp_avg_mean_sqrt = math.sqrt(exp_avg_sq_hat_sum / param_size)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                #Perform stable weight decay
                if group['weight_decay'] !=0:
                    
                    if group['variant'] == 'v4':
                        p.data.mul_(1 - group['weight_decay'] * group['lr'] / exp_avg_mean_sqrt)
                    elif group['variant'] == 'v5': #not working
                        p.data.mul_(1 - group['weight_decay'] * group['lr'] * state['S_t'])
                    elif group['variant'] == 'v6':
                        p.data.mul_(1 - group['weight_decay'] * group['lr'] * grad_norm_snr)
                    elif group['variant'] == 'v7':
                        p.data.mul_(1 - group['weight_decay'] * group['lr'] / (grad_norm_snr + group['eps']))
                    else:
                        p.mul_(1 - group['weight_decay'] * group['lr'])
                p.add_(-state['term_update'])

        return loss
