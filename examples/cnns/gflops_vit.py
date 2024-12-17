'''
Author: Unknown
Date: 2024-11-14 02:00:08
LastEditTime: 2024-11-14 04:52:03
LastEditors: Unknown
Description: 
FilePath: /Unknown/gflops_vit.py
'''
import time
import torch
from fvcore.nn import FlopCountAnalysis
import torch.nn as nn
import torch.optim as optim


# add args
import argparse
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--model', default='vit_s_16', type=str, help='model name')
parser.add_argument('--opt', default='Ours', type=str, help='optimizer name')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
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

assert args.model in ['vit_s_16', 'vit_h_14'], 'other model not supported in this script'

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(args.device)
# bs = BATCH_SIZE = 1
bs = BATCH_SIZE = args.batch_size
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

_model_params = {  # model parameters
    'num_classes': 1000,
    'image_size': 224 # manually set image_size for ViT
}
model = get_model(model_name, **_model_params) # get model from models/__init__.py
# model = get_model('vit_b_32', **_model_params) # get model from models/__init__.py
model = model.to(device)

input_tensor = torch.randn(bs, 3, 224, 224, device=device)
output = torch.rand_like(model(input_tensor), device=device)


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

opts = {
    'Ours': optimizers(model, 'SupEffAdaSNR-profiling', 1e-3, 5e-2),
    'Adam': optimizers(model, 'Adam-profiling', 1e-3, 5e-2),
    'AdamW': optimizers(model, 'AdamW-profiling', 1e-3, 5e-2),
    'SGD': optimizers(model, 'SGD-profiling', 1e-3, 5e-2),
    'Prodigy': optimizers(model, 'Prodigy-profiling', 1e-3, 5e-2),
    'Adam_mini': optimizers(model, 'AdamMini-profiling', 1e-3, 5e-2, model_name=model_name)
}



# Function to calculate FLOPs and G-FLOPS
def calculate_gflops(optimizer_name, optimizer):
    # Zero gradients
    optimizer.zero_grad()
    # Forward pass
    output = model(input_tensor)
    target = torch.randn_like(output)
    loss = criterion(output, target)

    # Backward pass
    loss.backward()
    if optimizer_name == 'Ours':
        optimizer.warmup_step()
    
    # Measure FLOPs for the backward pass
    flop_analysis =FlopCountAnalysis(model, input_tensor)
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


# # Calculate and display G-FLOPs for each optimizer
# for name, opt in opts.items():
#     for i in range(args.num_iters):
#         calculate_gflops(name, opt)

# Calculate and display G-FLOPs for each optimizer
opt_instance = opts[args.opt]
for i in range(args.num_iters):
    calculate_gflops(args.opt, opt_instance)