'''
Author: Unknown
Date: 2024-05-18 23:44:20
LastEditTime: 2024-05-18 23:46:43
LastEditors: Unknown
Description: partially copied from https://github.com/idstcv/ZenNAS/blob/main/global_utils.py
FilePath: /Unknown/models/PlainNet/global_utils.py
'''

def smart_round(x, base=None):
    if base is None:
        if x > 32 * 8:
            round_base = 32
        elif x > 16 * 8:
            round_base = 16
        else:
            round_base = 8
    else:
        round_base = base

    return max(round_base, round(x / float(round_base)) * round_base)
