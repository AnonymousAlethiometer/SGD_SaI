'''
Author: Unknown
Date: 2024-05-18 23:48:10
LastEditTime: 2024-05-18 23:50:10
LastEditors: Unknown
Description: test zennas 
FilePath: /Unknown/tests/test_zennas.py
'''
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ZenNet import get_ZenNet

def test_zennas():

    model = get_ZenNet('zennet_cifar10_model_size05M_res32', pretrained=False)
    print(model)

    model = get_ZenNet('zennet_cifar10_model_size1M_res32', pretrained=False)
    print(model)

    model = get_ZenNet('zennet_cifar10_model_size2M_res32', pretrained=False)
    print(model)

if __name__ == '__main__':
    test_zennas()
