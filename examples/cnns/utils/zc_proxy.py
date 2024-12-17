'''
Author: Unknown
Date: 2024-01-26 17:01:29
LastEditTime: 2024-08-13 11:21:36
LastEditors: Unknown
Description: 
FilePath: /GitHub/Unknown/utils/zc_proxy.py
'''
import torch
import torchvision


def calc_synflow(grads, model):
    # # calculate synflow score
    # -------- calc synflow with snr --------
    layer_synflow = []
    layer_snrsynflow = []
    for g, p in zip(grads, model.parameters()):
        s = torch.abs(g * p)
        
        # calculate std of gradients
        sigma = torch.std(s)
        sigma = torch.tensor(0.) if torch.isnan(sigma) else sigma  ###TODO: Notice, torch.std(torch.tensor([2.2]))==tensor(nan), np.std([2.2])==0.
    
        # calculate synflow score with its snr version
        syn = s.sum()
        syn_snr = syn / sigma if sigma != 0 else syn
        syn_snr = syn_snr.log1p()

        layer_synflow.append(syn)
        layer_snrsynflow.append(syn_snr)

    layer_synflow = torch.stack(layer_synflow)
    layer_snrsynflow = torch.stack(layer_snrsynflow)
    synflow_score = layer_synflow.sum().detach().item()
    snrsynflow_score = layer_snrsynflow.sum().detach().item()
    # -------- calc synflow with snr --------

    return synflow_score, snrsynflow_score


def calc_grad_norm(grads):
    '''
    This function also record and accumulate the grad_norm_std_score.
    '''

    # calculate grad_norm
    layer_grad_norm = []
    layer_snr_grad_norm = []
    layer_grad_norm_std = []
    layer_snr_grad_norm_log1p = []
    for g in grads:
        # calculate layer-wise std of gradients
        sigma = torch.std(g)
        sigma = torch.tensor(0., device=g.device) if torch.isnan(sigma) else sigma  #fix bug for diff devices.
        layer_grad_norm_std.append(sigma)

        # calculate layer-wise grad_norm
        grad_norm = g.norm()
        grad_norm_snr = (grad_norm / sigma) if sigma != 0 else grad_norm
        # grad_norm_snr = grad_norm_snr.log1p()
        grad_norm_snr_log1p = grad_norm_snr.log1p()
        grad_norm_snr = grad_norm_snr

        layer_grad_norm.append(grad_norm)
        layer_snr_grad_norm.append(grad_norm_snr)
        layer_snr_grad_norm_log1p.append(grad_norm_snr_log1p)

    layer_grad_norm = torch.stack(layer_grad_norm)
    layer_snr_grad_norm = torch.stack(layer_snr_grad_norm)
    layer_snr_grad_norm_log1p = torch.stack(layer_snr_grad_norm_log1p)
    layer_grad_norm_std = torch.stack(layer_grad_norm_std)
    grad_norm_score = layer_grad_norm.sum().detach().item()
    grad_norm_mean_score = layer_grad_norm.mean().detach().item()
    # snr_grad_norm_score = layer_snr_grad_norm.sum().detach().item()
    snr_grad_norm_score = layer_snr_grad_norm.sum().detach().item()
    snr_grad_norm_mean_score = layer_snr_grad_norm.mean().detach().item()
    snr_grad_norm_log1p_score = layer_snr_grad_norm_log1p.sum().detach().item()
    grad_norm_std_score = layer_grad_norm_std.sum().detach().item()

    return grad_norm_score, grad_norm_mean_score, snr_grad_norm_score, snr_grad_norm_mean_score, snr_grad_norm_log1p_score, grad_norm_std_score