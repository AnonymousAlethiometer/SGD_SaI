'''
Author: Unknown
Date: 2024-08-10 15:25:06
LastEditTime: 2024-12-14 18:19:46
LastEditors: Unknown
Description: 
v1.0.3 - remove warmup_step function for out-of-box usage, no need to call warmup_step function explicitly, friendly for distributed training, fix some potential bugs
v1.0.2 - add warmup_step function
FilePath: /SGD_SaI/sgd_sai/sgd_sai.py
'''
import torch
from torch.optim.optimizer import Optimizer

class SGD_sai(Optimizer):
    r"""

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): coefficients used for computing
            running averages of gradient (default: 0.9)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
    
    Typical Use:
    >>> optimizer = SGD_sai(model.parameters(), lr=lr, momentum=0.9, eps=1e-08, weight_decay=weight_decay)
    >>> for _ in range(steps):
    >>>     pred = model(input_ids)
    >>>     loss = loss_fn(pred, labels)
    >>>     loss.backward()
    >>>     optimizer.step()
    >>>     optimizer.zero_grad(set_to_none=True)

    """

    def __init__(self, params, lr=1e-2, momentum=0.9, eps=1e-8, weight_decay=1e-2):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum < 1.0:
            raise ValueError("Invalid momentum parameter value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, eps=eps, weight_decay=weight_decay)
        super(SGD_sai, self).__init__(params, defaults)
        self.has_warmup = False


    def __setstate__(self, state):
        super(SGD_sai, self).__setstate__(state)

    @torch.no_grad()
    def warmup_step(self, closure=None):
        '''
        Should be called after the first time backward of loss. Plase see the example above.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        '''
        print("-"*40)
        print('warmup in SGD_sai')
        print("-"*40)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                
                # calculate layer-wise grad_norm with std
                # sigma = torch.tensor(0.) if torch.isnan(sigma) else sigma
                # use nan_to_num to avoid nan value, and replace it with 0. nan happens when torch.tensor(4.).std() is called, the biased std will return nan
                sigma = d_p.std().nan_to_num()
                grad_norm = d_p.norm()
                # grad_norm_snr = (grad_norm / sigma) if sigma != 0 else grad_norm
                grad_norm_snr = (grad_norm / (sigma + group['eps']))
                param_state['gsnr'] = grad_norm_snr

        self.has_warmup = True
        return loss

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if not self.has_warmup:
            self.warmup_step(closure)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            lr = group['lr']
            momentum_factor = 1 - momentum
            decay_factor = 1 - weight_decay * lr

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                
                # if weight_decay != 0:
                #     d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                    # if len(param_state) == 0:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        # buf.mul_(momentum).add_(1 - momentum, d_p)
                        buf.mul_(momentum).add_(d_p, alpha=momentum_factor)
                # d_p = buf

                step_size = lr * param_state['gsnr'] 
                # p.data.mul_(1 - weight_decay * lr).add_(d_p, alpha=-step_size)
                p.data.mul_(decay_factor).add_(buf, alpha=-step_size)

        return loss
