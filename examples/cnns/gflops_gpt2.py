'''
Author: Unknown
Date: 2024-11-14 04:20:51
LastEditTime: 2024-11-14 04:50:21
LastEditors: Unknown
Description: 
FilePath: /Unknown/gflops_gpt2.py
'''
'''
Author: Unknown
Date: 2024-11-14 02:00:08
LastEditTime: 2024-11-14 04:18:17
LastEditors: Unknown
Description: 
FilePath: /Unknown/gflops.py
'''
import time
import torch
from fvcore.nn import FlopCountAnalysis
import torch.nn as nn
import torch.optim as optim


# add args
import argparse
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--model', default='gpt2_xl', type=str, help='model name')
parser.add_argument('--opt', default='Ours', type=str, help='optimizer name')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--context_length', default=256, type=int, help='context length')
parser.add_argument('--save_to', default='./gflops.csv', type=str, help='save to')
parser.add_argument('--device', default='cuda', type=str, help='device')
parser.add_argument('--num_iters', default=1, type=int, help='number of iterations')
args = parser.parse_args()


# # Define a simple model
# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         self.fc = nn.Linear(512,512)
#     def forward(self, x):
#         return self.fc(x)

assert args.model in ['gpt2_xl', ''], 'other model not supported in this script'


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(args.device)
# bs = BATCH_SIZE = 1
bs = BATCH_SIZE = args.batch_size
context_length = args.context_length
# model_name = 'vit_s_16'
# model_name = 'vit_h_14'
model_name = args.model
# SAVE_TO = './gflops.csv'
SAVE_TO = args.save_to
import os
if not os.path.exists(SAVE_TO):
    with open(SAVE_TO, 'w') as f:
        f.write('model_name,optimizer_name,gflops,total_flops,time\n')


from models import get_model

# _model_params = {  # model parameters
#     'num_classes': 1000,
#     'image_size': 224 # manually set image_size for ViT
# }
model = get_model('gpt2') # get model from models/__init__.py
# model = get_model('vit_b_32', **_model_params) # get model from models/__init__.py
model = model.to(device)



# context_length = 1024
input_tensor = torch.randint(0, 50256, (bs, context_length), dtype=torch.long, device=device)
output = torch.randint(0, 50256, (bs, context_length), dtype=torch.long, device=device)


# # Initialize model and input
# model = SimpleModel()
# input_tensor = torch.randn(1,512)
# output = model(input_tensor) # Forward pass to generate output

# # Define the loss function
# criterion = nn.MSELoss()
criterion = torch.nn.CrossEntropyLoss()
# Initialize optimizers
# optimizers ={
# "SGD": optim.SGD(model.parameters(), lr=0.01),
# "Adam": optim.Adam(model.parameters(),lr=0.01)
# }


from utils.optimizer import optimizers

# from models.gpt2.model import configure_optimizers_v2
optim_groups = model.configure_optimizers_v2(weight_decay=5e-2, device_type='cuda')

from optimizers import AdaSNR, AdaSNROLD, AdamW, Adam, SGD
from optimizers import Prodigy, Adam_mini, AdaSNR_SuperEfficient


def get_opt(OPTIMIZER):
    lr = 1e-3
    weight_decay = 5e-2

    if OPTIMIZER == "SupEffAdaSNR":
        optimizer = AdaSNR_SuperEfficient(optim_groups, lr=lr, beta=0.9, eps=1e-08, weight_decay=weight_decay, constant_wd=True)
    elif OPTIMIZER == "Adam":
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-2)
        optimizer = Adam(optim_groups, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
    elif OPTIMIZER == "AdamW":
        optimizer = AdamW(optim_groups, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
    elif OPTIMIZER == "SGD":
        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
        optimizer = SGD(optim_groups, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=False)
    elif OPTIMIZER == "Prodigy":
        optimizer = Prodigy(optim_groups, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
    elif OPTIMIZER == "Adam_mini":
        n_head = 25
        n_embd = 1600
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
        raise ValueError(f"Optimizer {OPTIMIZER} not supported.")
    return optimizer

opts = {
    'Ours': get_opt('SupEffAdaSNR'),
    'Adam': get_opt('Adam'),
    'AdamW': get_opt('AdamW'),
    'SGD': get_opt('SGD'),
    'Prodigy': get_opt('Prodigy'),
    'Adam_mini': get_opt('Adam_mini')
}


# Function to calculate FLOPs and G-FLOPS
def calculate_gflops(optimizer_name, optimizer):
    # Zero gradients
    optimizer.zero_grad()
    # Forward pass
    logits, loss = model(input_tensor, output)
    # target = torch.randn_like(output)
    # loss = criterion(output, target)

    # Backward pass
    loss.backward()
    if optimizer_name == 'Ours':
        optimizer.warmup_step()
    
    # Measure FLOPs for the backward pass
    flop_analysis = FlopCountAnalysis(model, input_tensor)
    total_flops = flop_analysis.total()

    # Measure the execution time of the optimizer step
    start_time = time.time()
    optimizer.step() # Update step
    end_time = time.time()
    execution_time = end_time - start_time

    # Calculate G-FLOPs
    gflops = total_flops /(execution_time * 1e9)
    print('-'*50)
    print(f"{optimizer_name} G-FLOPs: {gflops:.2f}")
    print('-'*50)
    with open(SAVE_TO, 'a') as f:
        f.write(f'{model_name},{optimizer_name},{gflops},{total_flops},{execution_time}\n')
    
# Calculate and display G-FLOPs for each optimizer
opt_instance = opts[args.opt]
for i in range(args.num_iters):
    calculate_gflops(args.opt, opt_instance)
