'''
Author: Unknown
Date: 2024-02-23 02:39:08
LastEditTime: 2024-03-04 11:06:40
LastEditors: Unknown
Description: https://github.com/VITA-Group/Sparsity-Win-Robust-Generalization/blob/main/Robust-Bird/utils/function.py
FilePath: /Unknown/utils/function.py
'''
import os
import time
import torch
import shutil
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from advertorch.attacks import LinfPGDAttack, L2PGDAttack
from advertorch.context import ctx_noparamgrad



#################### training epochs definition ##################################################
def train_epoch(train_loader, model, criterion, optimizer, epoch, args):
    # Standard Training with SGD
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    start = time.time()
    for i, (input, target) in enumerate(train_loader):

        input = input.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(input)
        loss = criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {3:.2f}'.format(
                    epoch, i, len(train_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def train_epoch_adv(train_loader, model, criterion, optimizer, epoch, args):
    # PGD
    losses = AverageMeter()
    top1 = AverageMeter()

    if args.norm == 'linf':
        adversary = LinfPGDAttack(
            model, loss_fn=criterion, eps=args.train_eps, nb_iter=args.train_step, eps_iter=args.train_gamma,
            rand_init=args.train_randinit, clip_min=0.0, clip_max=1.0, targeted=False
        )
    elif args.norm == 'l2':
        adversary = L2PGDAttack(
            model, loss_fn=criterion, eps=args.train_eps, nb_iter=args.train_step, eps_iter=args.train_gamma,
            rand_init=args.train_randinit, clip_min=0.0, clip_max=1.0, targeted=False
        )

    model.train()
    start = time.time()
    for i, (input, target) in enumerate(train_loader):

        input = input.cuda()
        target = target.cuda()

        #adv samples
        with ctx_noparamgrad(model):
            input_adv = adversary.perturb(input, target)

        # compute output
        output_adv = model(input_adv)
        loss = criterion(output_adv, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_adv.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {3:.2f}'.format(
                    epoch, i, len(train_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def clamp(X, l, u, cuda=True):
    if type(l) is not torch.Tensor:
        if cuda:
            l = torch.cuda.FloatTensor(1).fill_(l)
        else:
            l = torch.FloatTensor(1).fill_(l)
    if type(u) is not torch.Tensor:
        if cuda:
            u = torch.cuda.FloatTensor(1).fill_(u)
        else:
            u = torch.FloatTensor(1).fill_(u)
    return torch.max(torch.min(X, u), l)


def get_lr_schedule(lr_schedule_type, n_epochs, lr_max):
    if lr_schedule_type == 'cyclic':
        lr_schedule = lambda t: np.interp([t], [0, n_epochs * 2 // 5, n_epochs], [0, lr_max, 0])[0]
    elif lr_schedule_type == 'piecewise':
        def lr_schedule(t):
            if t >= 100:
                return lr_max / 10
            elif t >= 105:
                return lr_max / 100.
            elif t < 100:
                return lr_max
    else:
        raise ValueError('wrong lr_schedule_type')
    return lr_schedule


def model_eval(model, half_prec):
    model.eval()


def model_train(model, half_prec):
    model.train()


def get_uniform_delta(shape, eps, requires_grad=True):
    delta = torch.zeros(shape).cuda()
    delta.uniform_(-eps, eps)
    delta.requires_grad = requires_grad
    return delta

def sign(grad):
    grad_sign = torch.sign(grad)
    return grad_sign

def backward(loss, opt, half_prec):
    if half_prec:
        with amp.scale_loss(loss, opt) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()



iteration = 0
def train_epoch_fast(train_loader, model, criterion, optimizer, epoch, stage, args):
    global iteration
    train_loss, train_reg, train_acc, train_n, grad_norm_x, avg_delta_l2 = 0, 0, 0, 0, 0, 0
    eps = args.train_eps
    args.grad_align_cos_lambda = 0
    double_bp = True if args.grad_align_cos_lambda > 0 else False
    half_prec = False
    opt = optimizer
    lr_schedule = get_lr_schedule('cyclic', args.epochs1 if stage=='stage1' else args.epochs2 if stage=='stage2' else None, .2)
    loss_function = criterion
    for i, (X, y) in enumerate(train_loader):
        time_start_iter = time.time()
        # epoch=0 runs only for one iteration (to check the training stats at init)
        X, y = X.cuda(), y.cuda()
        lr = lr_schedule(epoch+ (i + 1) / len(train_loader))  # epoch - 1 since the 0th epoch is skipped
        #print('lr',lr)
        opt.param_groups[0].update(lr=lr)

        model_eval(model, False)
        delta = get_uniform_delta(X.shape, eps, requires_grad=True)

        X_adv = clamp(X + delta, 0, 1)
        output = model(X_adv)
        loss = F.cross_entropy(output, y)

        grad = torch.autograd.grad(loss, delta, create_graph=True if double_bp else False)[0]

        grad = grad.detach()

        argmax_delta = eps * sign(grad)

        n_alpha_warmup_epochs = 5
        if 'cifar10' in args.data:
            dataset_size = 50000
        if 'tiny' in args.data:
            dataset_size = 100000
        n_iterations_max_alpha = n_alpha_warmup_epochs * dataset_size // args.batch_size
        fgsm_alpha = 1.25 # maybe 1.25?
        delta.data = clamp(delta.data + fgsm_alpha * argmax_delta, -eps, eps)
        delta.data = clamp(X + delta.data, 0, 1) - X
        model_train(model, False)

        delta = delta.detach()

        output = model(X + delta)
        loss = loss_function(output, y)

        reg = torch.zeros(1).cuda()[0]  # for .item() to run correctly

        loss += reg

        opt.zero_grad()
        backward(loss, opt, half_prec)
        opt.step()

        train_loss += loss.item() * y.size(0)
        train_reg += reg.item() * y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_n += y.size(0)

        iteration += 1

    return train_acc / train_n * 100

#testing
def test(val_loader, model, criterion, args):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    start = time.time()
    for i, (input, target) in enumerate(val_loader):

        input = input.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % 50 == 0:
            end = time.time()
            print('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {2:.2f}'.format(
                    i, len(val_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('Standard Accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def test_adv(val_loader, model, criterion, args):
    """
    Run adversarial evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    if args.norm == 'linf':
        adversary = LinfPGDAttack(
            model, loss_fn=criterion, eps=args.test_eps, nb_iter=args.test_step, eps_iter=args.test_gamma,
            rand_init=args.test_randinit, clip_min=0.0, clip_max=1.0, targeted=False
        )
    elif args.norm == 'l2':
        adversary = L2PGDAttack(
            model, loss_fn=criterion, eps=args.test_eps, nb_iter=args.test_step, eps_iter=args.test_gamma,
            rand_init=args.test_randinit, clip_min=0.0, clip_max=1.0, targeted=False
        )

    model.eval()
    start = time.time()
    for i, (input, target) in enumerate(val_loader):

        input = input.cuda()
        target = target.cuda()

        #adv samples
        input_adv = adversary.perturb(input, target)
        # compute output
        with torch.no_grad():
            output = model(input_adv)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {2:.2f}'.format(
                    i, len(val_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('Robust Accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


#################### pruning definition ##################################################
import torch.nn.utils.prune as prune
def prune_unstructured(model, pct):
	params = []
	for m in model.modules():
		if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
			params.append((m, 'weight'))
	prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=pct)
	return model


#################### metrics definitions #################################################
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    #print(pred)
    #print(target)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


## misc #############
def save_checkpoint(state, is_SA_best, is_RA_best, save_path, filename='checkpoint.pth.tar'):

    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)

    if is_SA_best:
        shutil.copyfile(filepath, os.path.join(save_path, 'model_SA_best.pth.tar'))
    if is_RA_best:
        shutil.copyfile(filepath, os.path.join(save_path, 'model_RA_best.pth.tar'))

