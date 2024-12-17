'''
Author: Unknown
Date: 2024-02-16 18:49:29
LastEditTime: 2024-11-18 21:40:29
LastEditors: Unknown
Description: 
    This module provides various optimizer implementations for Unknown.
        copied from:
        https://github.com/zeke-xie/stable-weight-decay-regularization/
    Available Optimizers:
    - Adai: Adaptive learning rate optimizer.
    - AdaiS: Adaptive learning rate optimizer with scale factor.
    - AdamS: Adaptive Moment Estimation optimizer with scale factor.
    - SGDS: Stochastic Gradient Descent optimizer with scale factor.
FilePath: /Unknown/optimizers/__init__.py
'''


from .adai import Adai
from .adais import AdaiS
from .adams import AdamS
from .sgds import SGDS

from .adamw import AdamW
from .adam import Adam
from .sgd import SGD

from .prodigy import Prodigy
# from .adam_mini_v2 import Adam_mini
# from .sgd_boost import SGD_boost
from .sgd_sai import SGD_sai

# Clean up unused imports
del adai
del adais
del adams
del sgds
del adamw
del adam
del sgd


del prodigy
# del adam_mini_v2
# del sgd_boost
del sgd_sai
