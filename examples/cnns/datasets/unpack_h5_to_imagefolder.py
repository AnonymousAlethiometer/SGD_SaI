'''
Author: Unknown
Date: 2024-04-10 01:35:31
LastEditTime: 2024-04-10 02:40:13
LastEditors: Unknown
Description: Try to read ImageNet1k h5 file and transform it to ImageFolder like datasets
FilePath: /Unknown/datasets/unpack_h5_to_imagefolder.py
'''
import h5py
import numpy as np
from PIL import Image
import time
import os
from tqdm import tqdm

pth_from = './ImageNet1k/imagenet-train-256.h5'
pth_to = './imagenet1k_imagefolder/train/'

# pth_from = './ImageNet1k/imagenet-val-256.h5'
# pth_to = './imagenet1k_imagefolder/val/'

if not os.path.exists(pth_to):
    os.makedirs(pth_to)

# for a folder structure like ImageFolder
# root/dog/xxx.png
# root/dog/xxy.png
# root/dog/[...]/xxz.png
# root/cat/123.png
# root/cat/nsdf3.png
# root/cat/[...]/asd932_.png
with h5py.File(pth_from, 'r') as h5_file:
    for index in tqdm(range(len(h5_file))):

        record = h5_file[str(index)]

        x = Image.fromarray(record['data'][()])
        y = record['target'][()]

        pth = os.path.join(pth_to, str(y))
        if not os.path.exists(pth):
            os.makedirs(pth)
        
        x.save(os.path.join(pth, f'{index:07d}.jpg'))



