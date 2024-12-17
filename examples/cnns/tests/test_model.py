'''
Author: Unknown
Date: 2024-11-12 00:29:46
LastEditTime: 2024-11-12 01:00:21
LastEditors: Unknown
Description: 
FilePath: /Unknown/tests/test_model.py
'''

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_model
import torch



model = get_model('gpt2') # get model from models/__init__.py
# model = get_model('vit_b_32', **_model_params) # get model from models/__init__.py
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# from typing import Long
x = torch.randint(0, 50256, (1, 1024), dtype=torch.long, device=device)
y = torch.randint(0, 50256, (1, 1024), dtype=torch.long, device=device)
print(x, y)
logits, loss = model(x, y)
print(logits.shape, loss)
# labels = torch.rand_like(model(inputs), device=device)