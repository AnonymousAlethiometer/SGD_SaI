'''
Author: Unknown
Date: 2024-03-06 11:06:54
LastEditTime: 2024-03-06 11:14:17
LastEditors: Unknown
Description: 
FilePath: /Unknown/tests/test_randomness.py
'''

import numpy as np
import time
import random

if __name__ == '__main__':

    print('This is a test file for randomness.')
    
    print(np.random.rand())
    print(np.random.rand())
    print('[0] set seed to 0.')
    np.random.seed(0)
    print(np.random.rand())
    print(np.random.rand())
    print(np.random.rand())
    print('[1] set seed to 0 again.')
    np.random.seed(0)
    a = 0
    for i in range(10000000):
        a += 1
    print("finish adding of a: ", a)
    print(np.random.rand())
    print(np.random.rand())
    print(np.random.rand())
    print('[2] set seed to 0 again.')
    np.random.seed(0)
    print("begin sleeping for 1 second.")
    time.sleep(1)
    print("finish sleeping for 1 second.")
    print("using random.random :",random.random())
    print(np.random.rand())
    print(np.random.rand())
    print(np.random.rand())

    



