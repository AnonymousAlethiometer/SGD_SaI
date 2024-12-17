import math
import torch
from torch.optim.optimizer import Optimizer

from utils.shared_variables import SHARED_VARS

class AdamS(Optimizer):
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
                 weight_decay=1e-4, amsgrad=False, eq1=False):
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
                        weight_decay=weight_decay, amsgrad=amsgrad, eq1=eq1)
        super(AdamS, self).__init__(params, defaults)

        args = SHARED_VARS.get('args', None)
        self.args = args
        if args is not None:
            self.wd_ratio_monitor = args.wd_ratio_monitor
        else:
            self.wd_ratio_monitor = False
        self.wd_ratio_norm_coeff = lr

    def __setstate__(self, state):
        super(AdamS, self).__setstate__(state)
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
        exp_avg_sq_hat_sum = 0.
        # record wd_ratio
        wd_ratio = 1.
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_size += p.numel()

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamS does not support sparse gradients')
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

                beta1, beta2 = group['betas']
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    exp_avg_sq_hat = max_exp_avg_sq / bias_correction2
                else:
                    exp_avg_sq_hat = exp_avg_sq / bias_correction2
                
                exp_avg_sq_hat_sum += exp_avg_sq_hat.sum()

        # Calculate the sqrt of the mean of all elements in exp_avg_sq_hat  
        exp_avg_mean_sqrt = math.sqrt(exp_avg_sq_hat_sum / param_size)
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                #Perform stable weight decay
                if group['weight_decay'] !=0:
                    if group['eq1']:
                        p.data.mul_(1 - group['weight_decay'])
                    else:
                        p.data.mul_(1 - group['weight_decay'] * group['lr'] / exp_avg_mean_sqrt)
                    
                    
                    # p.data.mul_(1 - group['weight_decay'] * group['lr'])

                beta1, beta2 = group['betas']
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    exp_avg_sq_hat = max_exp_avg_sq / bias_correction2
                else:
                    exp_avg_sq_hat = exp_avg_sq / bias_correction2
                
                denom = exp_avg_sq_hat.sqrt().add(group['eps'])
                
                step_size = group['lr'] / bias_correction1 
                p.addcdiv_(exp_avg, denom, value= - step_size)
                
        
        # make results available to other parts of the code
        if self.wd_ratio_monitor:
            # record wd ratio and store it in the tensorboard logs
            wd_ratio = 1. / exp_avg_mean_sqrt
            # print('wd_ratio:', wd_ratio)
            SHARED_VARS['wd_ratio'] = wd_ratio
            SHARED_VARS['wd_ratio_normalized'] = wd_ratio * self.wd_ratio_norm_coeff
        
        return loss
