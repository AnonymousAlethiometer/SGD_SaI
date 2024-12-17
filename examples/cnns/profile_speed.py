'''
Author: Unknown
Date: 2024-11-18 21:46:12
LastEditTime: 2024-11-21 00:32:42
LastEditors: Unknown
Description: https://pytorch.org/blog/understanding-gpu-memory-1/
FilePath: /Unknown/profile_speed.py
'''


# (c) Meta Platforms, Inc. and affiliates. 
import logging
import socket
from datetime import datetime, timedelta
import numpy as np

# from utils.optimizer import optimizers
from optimizers import AdamW, Adam, SGD, Prodigy, SGD_sai
from adam_mini import Adam_mini

import torch
import time

from torch.autograd.profiler import record_function
from torchvision import models

from models import get_model


# OPTIMIZER="AdaSNR-v2"
# OPTIMIZER="Adam"
# OPTIMIZER="AdamW"
# OPTIMIZER="SGD"
# OPTIMIZER="Prodigy"
# OPTIMIZER="Adam_mini"
OPTIMIZER="SGD_sai"
COMMON_STORE_PATH = "./profiles"
COMMON_PREFIX = "profile_"



logging.basicConfig(
    format="%(levelname)s:%(asctime)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

def trace_handler(prof: torch.profiler.profile):
    # Prefix for file names.
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"{COMMON_PREFIX}_{host_name}_{timestamp}"

    # Construct the trace file.
    prof.export_chrome_trace(f"{COMMON_STORE_PATH}/{file_prefix}.json.gz")

    # Construct the memory timeline file.
    # prof.export_memory_timeline(f"{COMMON_STORE_PATH}/{file_prefix}.html", device="cuda:0")
    prof.export_memory_timeline(f"{COMMON_STORE_PATH}/{file_prefix}.json", device="cuda:0")
    # prof.export_memory_timeline(f"{COMMON_STORE_PATH}/{file_prefix}.pdf", device="cuda:0")

    digest_results(f"{COMMON_STORE_PATH}/{file_prefix}.json")


def digest_results(json_name):
    import json
    dt = { f'memory_usage_{i}': [] for i in range(9) }
    dt = { 'timestamp': [], **dt }
    with open(json_name, 'r') as f:
        data = json.load(f)

        assert len(data) == 2
        # load the memory timeline data, and calculate the memory usage and the runtime of total training process
        runtime_times = data[0]  # list of timestamps of each tick
        memory_usage = data[1]  # list of memory usage of each tick
        # change it into csv file
        for i in range(len(runtime_times)):
            dt['timestamp'].append(runtime_times[i])
            for j in range(9):
                dt[f'memory_usage_{j}'].append(memory_usage[i][j])
        
    import pandas as pd
    df = pd.DataFrame(dt)
    cols = df.columns.tolist()[1:]
    df['total_memory_usage'] = df[cols].sum(axis=1)
    df['readable_total'] = df['total_memory_usage'].apply(lambda x: f"{x/1024**3:.4f} GB")

    # rename cols 
    # /home/Unknown/miniconda3/envs/torch_profile/lib/python3.11/site-packages/torch/profiler/_memory_profiler.py #method: export_memory_timeline_html
    from torch.profiler._memory_profiler import _CATEGORY_TO_COLORS
    new_cols = ["Unknown" if i is None else i.name for i in _CATEGORY_TO_COLORS]
    total_cols = df.columns.tolist()[0:2] + new_cols + ['total_memory_usage', 'readable_total']
    df.columns = total_cols

    # add step index column, the step can be seperated when the GRADIENT value first becomes 0 for each step, the total steps is num_iters, and the step index is the index of the first 0 value
    step=-1
    previous_gradient_value = -1
    for row in df.iterrows():
        row = row[1]
        current_gradient_value = row['GRADIENT']
        if current_gradient_value == 0 and previous_gradient_value != 0:
            step += 1
        df.loc[row.name, 'step_index'] = step
        previous_gradient_value = current_gradient_value

    df.to_csv(json_name.replace('.json', '.csv'), index=False)

    # record total runtime and the max memory usage
    from datetime import datetime, timedelta
    timestamps = df['timestamp']
    # total_runtime = timedelta(microseconds=float(timestamps[len(timestamps)-1] - timestamps[0]))
    total_runtime = timestamps[len(timestamps)-1] - timestamps[0]  #microseconds, 1e-6
    max_memory = df['total_memory_usage'].max()
    readable_max_memory = f"{max_memory/1024**3:.4f} GB"
    state_memory = df['OPTIMIZER_STATE'].max()
    readable_state_memory = f"{state_memory/1024**3:.4f} GB"
    exp_name = json_name

    with open(json_name.replace('.json', '.txt'), 'w') as f:
        f.write(f"Total runtime: {total_runtime} microseconds\n")
        f.write(f"Max memory usage: {readable_max_memory}\n")
    
    global COMMON_STORE_PATH
    global OPTIMIZER
    global args
    if not os.path.exists(f"{COMMON_STORE_PATH}/summary.csv"):
        with open(f"{COMMON_STORE_PATH}/summary.csv", 'w') as f:
            f.write("optimizer,num_iters,batch_size,total_runtime,step0_runtime,steprest_runtime,max_memory,state_memory,readable_max_memory,readable_state_memory,exp_name,whole_start_time,whole_end_time,step_1_start_time\n")
    with open(f"{COMMON_STORE_PATH}/summary.csv", 'a') as f:
        f.write(f"{OPTIMIZER},{args.num_iters},{args.batch_size},{total_runtime},{None},{None},{max_memory},{state_memory},{readable_max_memory},{readable_state_memory},{exp_name},{None},{None},{None}\n")


def init_model(model_name='resnet50'):
    if model_name == 'resnet50':
        model = models.resnet50()
    elif model_name == 'vit_s_16' or model_name == 'vit_b_32' or model_name == 'vit_l_16' or model_name == 'vit_h_14':
        _model_params = {  # model parameters
            'num_classes': 1000,
            'image_size': 224 # manually set image_size for ViT
        }
        model = get_model(model_name, **_model_params) # get model from models/__init__.py
    elif model_name == 'gpt2':
        model = get_model(model_name) # get model from models/__init__.py
    else:
        raise ValueError(f"Model {model_name} not supported.")
    return model

def init_inputs(model_name='resnet50', device='cuda:0', bs=1, model=None, context_length=1024):
    if model_name == 'resnet50':
        inputs = torch.randn(bs, 3, 224, 224, device=device)
        labels = torch.rand_like(model(inputs))
    elif model_name == 'vit_s_16' or model_name == 'vit_b_32' or model_name == 'vit_l_16' or model_name == 'vit_h_14':
        inputs = torch.randn(bs, 3, 224, 224, device=device)
        labels = torch.rand_like(model(inputs))
    elif model_name == 'gpt2':
        # context_length = 1024
        inputs = torch.randint(0, 50256, (bs, context_length), dtype=torch.long, device=device)
        labels = torch.randint(0, 50256, (bs, context_length), dtype=torch.long, device=device)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    return inputs, labels

def init_optimizer(model_name='resnet50', opt='SupEffAdaSNR', model=None):
    optim_groups = model.configure_optimizers_v2(weight_decay=5e-2, device_type='cuda') if 'gpt2' in model_name else model.parameters()

    lr = 1e-3
    weight_decay = 5e-2
    
    if opt == "SGD_sai":
        optimizer = SGD_sai(optim_groups, lr=lr, momentum=0.9, eps=1e-08, weight_decay=weight_decay)
    elif opt == "Adam":
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-2)
        optimizer = Adam(optim_groups, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
    elif opt == "AdamW":
        optimizer = AdamW(optim_groups, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
    elif opt == "SGD":
        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
        optimizer = SGD(optim_groups, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=False)
    elif opt == "Prodigy":
        optimizer = Prodigy(optim_groups, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
    elif opt == "Adam_mini":
        if model_name == 'gpt2':
            n_head = 25
            n_embd = 1600
        if model_name == 'vit_s_16':
            n_head=6
            n_embd=384
        elif model_name == 'vit_b_32':
            n_head=12
            n_embd=768
        elif model_name == 'vit_l_16':
            n_head=16
            n_embd=1024
        elif model_name == 'vit_h_14':
            n_head=16
            n_embd=1280

        optimizer = Adam_mini(
            named_parameters=model.named_parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
            model_sharding=False,
            dim=n_embd,
            n_heads=n_head
        )
    else:
        raise ValueError(f"Optimizer {opt} not supported.")


    return optimizer

def common_proc(model_name='resnet50', device='cuda:0', opt='SupEffAdaSNR', bs=1, context_length=None, num_iters=5, after_warmup=False, fn_torch_profiler=None, fn_time_profiler=None, args=None):
    model = init_model(model_name)
    model = model.to(device)
    inputs, labels = init_inputs(model_name, device, bs, model, context_length)
    optimizer = init_optimizer(model_name, opt, model)

    global OPTIMIZER
    OPTIMIZER = opt
    global COMMON_PREFIX
    COMMON_PREFIX = f"{OPTIMIZER}-n_iters_{num_iters}-bs_{bs}-contextlen_{context_length}"
    if after_warmup:
        global COMMON_STORE_PATH
        import os
        COMMON_STORE_PATH = f"{COMMON_STORE_PATH}/after_warmup"
        if not os.path.exists(COMMON_STORE_PATH):
            os.makedirs(COMMON_STORE_PATH)
    
    print('-'*50)
    print(f"Optimizer: {OPTIMIZER}")
    print(f"Number of iterations: {num_iters}")
    print(f"Batch size: {bs}")
    print('-'*50)

    if fn_torch_profiler:
        fn_torch_profiler(num_iters=num_iters, optimizer=optimizer, inputs=inputs, labels=labels, model=model, args=args)
    
    if fn_time_profiler:
        fn_time_profiler(num_iters=num_iters, optimizer=optimizer, inputs=inputs, labels=labels, model=model, args=args)


def torch_profiler_gpt2(num_iters=5, optimizer=None, inputs=None, labels=None, model=None, args=None):
    loss_fn = torch.nn.CrossEntropyLoss()
    
    global OPTIMIZER
    need_warmup = (OPTIMIZER == "SGD_sai")
    # the configuration of schedule is to profile the whole training process, repeat=1 means only 1 cycle. active means the number of iterations to profile.
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=num_iters, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=trace_handler,
    ) as prof:
        for _ in range(num_iters):
            with record_function("## forward ##"):
                logits, loss = model(inputs, labels)

            with record_function("## backward ##"):
                loss.backward()
                if need_warmup:
                    optimizer.warmup_step()
                    need_warmup = False

            with record_function("## optimizer ##"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            prof.step()


def time_profiler_gpt2(num_iters=5, optimizer=None, inputs=None, labels=None, model=None, args=None):
    loss_fn = torch.nn.CrossEntropyLoss()

    global OPTIMIZER
    need_warmup = (OPTIMIZER == "SGD_sai")
    exec_times = []
    warmup_time = 0
    for _ in range(num_iters):
        logits, loss = model(inputs, labels)

        loss.backward()
        if need_warmup:
            start_time = time.time()
            optimizer.warmup_step()
            end_time = time.time()
            warmup_time = end_time - start_time
            need_warmup = False

        # Measure the execution time of the optimizer step
        start_time = time.time()
        optimizer.step() # Update step
        end_time = time.time()
        execution_time = end_time - start_time
        exec_times.append(execution_time)

        optimizer.zero_grad(set_to_none=True)
    
    _time_write_exec_times(exec_times, warmup_time, f"{COMMON_STORE_PATH}/execution_times.csv", args)


def torch_profiler_vit(num_iters=5, optimizer=None, inputs=None, labels=None, model=None, args=None):
    loss_fn = torch.nn.CrossEntropyLoss()
    
    global OPTIMIZER
    need_warmup = (OPTIMIZER == "SGD_sai")
    # the configuration of schedule is to profile the whole training process, repeat=1 means only 1 cycle. active means the number of iterations to profile.
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=num_iters, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=trace_handler,
    ) as prof:
        for _ in range(num_iters):
            with record_function("## forward ##"):
                pred = model(inputs)

            with record_function("## backward ##"):
                loss_fn(pred, labels).backward()
                if need_warmup:
                    optimizer.warmup_step()
                    need_warmup = False

            with record_function("## optimizer ##"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            prof.step()


def time_profiler_vit(num_iters=5, optimizer=None, inputs=None, labels=None, model=None, args=None):
    loss_fn = torch.nn.CrossEntropyLoss()

    global OPTIMIZER
    need_warmup = (OPTIMIZER == "SGD_sai")
    exec_times = []
    warmup_time = 0
    for _ in range(num_iters):
        pred = model(inputs)

        loss_fn(pred, labels).backward()
        if need_warmup:
            start_time = time.time()
            optimizer.warmup_step()
            end_time = time.time()
            warmup_time = end_time - start_time
            need_warmup = False

        # Measure the execution time of the optimizer step
        start_time = time.time()
        optimizer.step() # Update step
        end_time = time.time()
        execution_time = end_time - start_time
        exec_times.append(execution_time)

        optimizer.zero_grad(set_to_none=True)
    
    _time_write_exec_times(exec_times, warmup_time, f"{COMMON_STORE_PATH}/execution_times.csv", args)

def _time_write_exec_times(exec_times, warmup_time=0., path=None, args=None):
    opt_name = args.optimizer
    model_name = args.model_name
    bs = args.batch_size
    num_iters = args.num_iters
    context_length = args.context_length
    
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write(f"model_name,optimizer,batch_size,num_iters,context_length,min_time,avg_time,std_time,warmup_time,execution_times\n")
    with open(path, 'a') as f:
        f.write(f"{model_name},{opt_name},{bs},{num_iters},{context_length},{min(exec_times)},{sum(exec_times)/len(exec_times)},{np.std(exec_times)},{warmup_time},\"{str(exec_times)}\"\n")



if __name__ == "__main__":
    # # Warm up
    # run_resnet50()
    # # Run the resnet50 model
    # run_resnet50()

    # add command line argument to specify the number of iterations and the optimizer
    import argparse
    parser = argparse.ArgumentParser(description='Profile the training process')
    parser.add_argument('--model_name', type=str, default='vit_s_16', help='Model name to profile')
    parser.add_argument('--num_iters', type=int, default=5, help='Number of iterations to profile')
    parser.add_argument('--optimizer', type=str, default='AdaSNR-v2', help='Optimizer to profile')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--context_length', type=int, default=1024, help='Token length for GPT2')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model')
    parser.add_argument('--root_path', type=str, default='./profiles', help='Root path to store the profiling results')
    args = parser.parse_args()

    COMMON_STORE_PATH = args.root_path
    import os
    if not os.path.exists(COMMON_STORE_PATH):
        os.makedirs(COMMON_STORE_PATH)

    if args.model_name.startswith('vit'):
        # Warm up
        common_proc(model_name=args.model_name, device=args.device, opt=args.optimizer, bs=args.batch_size, num_iters=args.num_iters, after_warmup=False, fn_torch_profiler=torch_profiler_vit, fn_time_profiler=time_profiler_vit, args=args)
        # Run the vit model
        common_proc(model_name=args.model_name, device=args.device, opt=args.optimizer, bs=args.batch_size, num_iters=args.num_iters, after_warmup=True, fn_torch_profiler=torch_profiler_vit, fn_time_profiler=time_profiler_vit, args=args)

    if args.model_name.startswith('gpt2'):
        # Warm up
        common_proc(model_name=args.model_name, device=args.device, opt=args.optimizer, bs=args.batch_size, context_length=args.context_length, num_iters=args.num_iters, after_warmup=False, fn_torch_profiler=torch_profiler_gpt2, fn_time_profiler=time_profiler_gpt2, args=args)
        # Run the gpt2 model
        common_proc(model_name=args.model_name, device=args.device, opt=args.optimizer, bs=args.batch_size, context_length=args.context_length, num_iters=args.num_iters, after_warmup=True, fn_torch_profiler=torch_profiler_gpt2, fn_time_profiler=time_profiler_gpt2, args=args)
    
    print("Profiling finished.")