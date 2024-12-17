'''
Author: Unknown
Date: 2024-02-16 23:38:52
LastEditTime: 2024-11-18 21:43:32
LastEditors: Unknown
Description: generate optimizer configured as SWD did.
FilePath: /Unknown/utils/optimizer.py
'''
import os
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from optimizers import Adai, AdaiS, AdamS, SGDS
from optimizers import AdamW, Adam, SGD
from optimizers import Prodigy, SGD_sai
from adam_mini import Adam_mini  ## pip install adam-mini


def optimizers(net, opti_name, lr, weight_decay, **kwargs):
    if opti_name == 'VanillaSGD':
        return optim.SGD(net.parameters(), lr=lr, momentum=0, weight_decay=weight_decay)
    elif opti_name == 'SGD':
        return optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=False)
    elif opti_name == 'SGDS':
        return SGDS(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=False)
    elif opti_name == 'Adam':
        return optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
    elif opti_name == 'AMSGrad':
        return optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay,amsgrad=True)
    elif opti_name == 'AdamW-eq1':
        return AdamS(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False, eq1=True)
    elif opti_name == 'AdamW-eq2' or opti_name == 'AdamW-LH' or opti_name == 'AdamW':
        return optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay/lr)
    elif opti_name == 'AdamS':
        return AdamS(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
    elif opti_name == 'Adai':
        return Adai(net.parameters(), lr=lr, betas=(0.1, 0.99), eps=1e-03, weight_decay=weight_decay)
    elif opti_name == 'AdaiS':
        return AdaiS(net.parameters(), lr=lr, betas=(0.1, 0.99), eps=1e-03, weight_decay=weight_decay)
    # ------- grid search -------
    elif opti_name == 'Adam-gs':
        return optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
    elif opti_name == 'AdamW-gs':
        return optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
    elif opti_name == 'AdamS-gs':
        return AdamS(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
    elif opti_name == 'SGD-gs':
        return optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=False)
    elif opti_name == 'SGDS-gs':
        return SGDS(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=False)
    elif opti_name == 'NSGD-gs':  # sgd with nesterov, not compared in cifar10/cifar100 datasets, only used in SSS/TSS
        return optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    # ------- grid search -------

    # ------- profiling the optimizers -------
    elif opti_name == 'AdamW-profiling':
        return AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
    elif opti_name == 'Adam-profiling':
        return Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
    elif opti_name == 'SGD-profiling':
        return SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=False)
    elif opti_name == 'AdamMini-profiling':
        if True:
            parameters = list(net.named_parameters())
            opt_args = {
                'lr': lr,
                'betas': (0.9, 0.999),
                'eps': 1e-08,
                'weight_decay': weight_decay,
            }
            model_name = kwargs.get('model_name', None)
            if model_name == 'vit_s_16':
                opt_args['n_heads']=6
                opt_args['dim']=384
            elif model_name == 'vit_b_32':
                opt_args['n_heads']=12
                opt_args['dim']=768
            elif model_name == 'vit_l_16':
                opt_args['n_heads']=16
                opt_args['dim']=1024
            elif model_name == 'vit_h_14':
                opt_args['n_heads']=16
                opt_args['dim']=1280
            # if 'vit_s_16':
            #     opt_args['n_heads']=6
            #     opt_args['dim']=384
            # if 'vit_b_32':
            #     opt_args['n_heads']=12
            #     opt_args['dim']=768
            optimizer = Adam_mini(parameters, **opt_args)
            optimizer.output_names.add('head.weight')
            optimizer.wqk_names.add('qkv.weight')
            return optimizer
        else:
            return Adam_mini(net.named_parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
    elif opti_name == 'Prodigy-profiling':
        return Prodigy(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
    elif opti_name == 'SGD-sai-profiling':
        return SGD_sai(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=False)
    # ------- profiling the optimizers -------
    else:
        raise('Unspecified optimizer.')


def record_optimizer_info(optimizer, writer: SummaryWriter = None, current_step: int=0, current_epoch: int=0, extra_save_pth: str=None):
    idx = 0
    grad_snrs = []
    gard_norms = []
    if type(optimizer) in [SGD_sai]:
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            writer.add_scalar('lr', lr, current_step)
            for p in param_group['params']:
                param_name = f'param_{idx}'
                idx += 1
                state = optimizer.state[p]
                if 'wd_ratio' in state:
                    writer.add_scalar(f'wd_ratio/{param_name}', state['wd_ratio'], current_step)
                if 'norm' in state:
                    writer.add_scalar(f'grad_norm/{param_name}', state['norm'], current_step)
                    writer.add_scalar(f'grad_norm_snr/{param_name}', state['norm_snr'], current_step)
                    writer.add_scalar(f'grad_norm_std/{param_name}', state['norm_std'], current_step)
                    effective_lr = lr * state['norm_snr']
                    writer.add_scalar(f'effective_lr/{param_name}', effective_lr, current_step)

                    save_pth = os.path.join(extra_save_pth, f'grad_norm_{param_name}.txt')
                    if not os.path.exists(save_pth):
                        with open(save_pth, 'w') as f:
                            f.write('epoch,step,norm,norm_snr,norm_std,lr,effective_lr\n')
                    with open(os.path.join(extra_save_pth, f'grad_norm_{param_name}.txt'), 'a') as f:
                        f.write(f'{current_epoch},{current_step},{state["norm"]},{state["norm_snr"]},{state["norm_std"]},{lr},{effective_lr}\n')
                    
                    grad_snrs.append(state['norm_snr'])
                    gard_norms.append(state['norm'])
    
    if len(grad_snrs) > 0:
        writer.add_scalar('grad_norm_snr_mean', sum(grad_snrs) / len(grad_snrs), current_step)
        writer.add_scalar('grad_norm_snr_std', sum([(x - sum(grad_snrs) / len(grad_snrs))**2 for x in grad_snrs]) / len(grad_snrs), current_step)
        writer.add_scalar('grad_norm_mean', sum(gard_norms) / len(gard_norms), current_step)


def record_gradient_histgram(optimizer, writer: SummaryWriter = None, current_step: int=0, extra_save_pth: str=None, gradient_distribution: list=None):
    '''
    The absolute gradient histogram of the Transformers during the training (stacked along the y-axis). 
    X-axis is absolute value in the log scale and the height is the frequency.
    Store this information in both a text file and a list.
    '''
    
    grads = [p.grad for group in optimizer.param_groups for p in group['params'] if p.grad is not None]

    if len(grads) == 0:
        return
    
    # calculate the log scale of the absolute value of the gradients in this step
    grad_abs = [grad.detach().abs().log().cpu().numpy().flatten() for grad in grads]
    # flatten the list and convert to numpy array
    grad_abs = np.concatenate(grad_abs)

    # store the gradient distribution in the list, accumulate the frequency and store the log scale of the absolute value
    writer.add_histogram('gradient_distribution', grad_abs, current_step)

    # store the gradient distribution in the text file
    if gradient_distribution is None:
        gradient_distribution = []
    gradient_distribution.append(grad_abs)
    with open(os.path.join(extra_save_pth, 'gradient_distribution.txt'), 'a') as f:
        f.write(','.join([str(g) for g in grad_abs]) + '\n')
    print(grad_abs.shape)
