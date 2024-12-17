'''
Author: Unknown
Date: 2024-11-11 23:08:05
LastEditTime: 2024-11-18 21:48:08
LastEditors: Unknown
Description: 
FilePath: /Unknown/models/gpt2/get_model.py
'''
import argparse
import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import json
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from .model import GPTConfig, GPT
from torch.utils.tensorboard import SummaryWriter
# from adam_mini import Adam_mini
# #import ipdb

# import logger
# import io_utils
# import torch_optimizer as optim




def gpt2_model():
    # ipdb.set_trace()

    # -----------------------------------------------------------------------------
    # default config values designed to train a gpt2 (124M) on OpenWebText
    # I/O
    out_dir = 'out'
    resume_dir = None
    eval_interval = 1000
    ckpt_interval = 1000
    log_interval = 1
    eval_iters = 200
    eval_only = False # if True, script exits right after the first eval
    init_from = 'scratch'
    load_iter = 0
    # data
    dataset = 'openwebtext' 
    gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
    batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size = 1024
    # model
    # n_layer = 12  #gpt2 small
    # n_head = 12
    # n_embd = 768
    n_layer = 48  #gpt2 xl also known as gpt2 1.5B
    n_head = 25
    n_embd = 1600
    dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    bias = False # do we use bias inside LayerNorm and Linear layers?
    #optimizer
    learning_rate = 6e-4 # max learning rate
    max_iters = 600000 # total number of training iterations
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    epsilon = 1e-8
    grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True # whether to decay the learning rate
    warmup_iters = 2000 # how many steps to warm up for
    lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
    min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    seed = 1337
    comment = 'none'
    algorithm = 'adam_mini'
    flash_attn = True
    # DDP settings
    backend = 'nccl' # 'nccl', 'gloo', etc.
    # system
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    # dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    # print('current dtype', dtype)

    # save_dir = 'log_gpt2/'+comment


    # # -----------------------------------------------------------------------------
    # config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    # exec(open('configurator.py').read()) # overrides from command line or config file
    # config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # # -----------------------------------------------------------------------------


    # os.makedirs(save_dir, exist_ok = True)
    # writer = SummaryWriter(save_dir)

    #  .... 

    # poor man's data loader
    data_dir = os.path.join('data', dataset)


    # train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    # val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    # def get_batch(split):
    #     data = train_data if split == 'train' else val_data
    #     ix = torch.randint(len(data) - block_size, (batch_size,))
    #     x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    #     y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    #     if device_type == 'cuda':
    #         # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
    #         x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    #     else:
    #         x, y = x.to(device), y.to(device)
    #     return x, y

    # # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    # iter_num = 0



    #load_iter = int(os.environ.get("LOAD_ITER"))
    print('load_iter = ', load_iter, 'loading ..', load_iter)

    if load_iter == 0:
        init_from = 'scratch'
    else: 
        init_from = 'resume'


    # attempt to derive vocab_size from the dataset
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    # model init
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                    bias=bias, vocab_size=None, dropout=dropout, flash_attn = flash_attn, device = device) # start with model_args from command line
    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size # so that the checkpoint will have the right value
    # model.to(device)
    
    return model