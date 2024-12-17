'''
Author: Unknown
Date: 2023-04-07 18:48:20
LastEditTime: 2024-04-05 21:24:51
LastEditors: Unknown
Description: h5py_dataset file from foresight/h5py_dataset.py
FilePath: /Unknown/datasets/h5py_dataset.py
'''
# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import h5py
import numpy as np
from PIL import Image
import time
import threading


import torch
from torch.utils.data import Dataset, DataLoader

class H5Dataset(Dataset):
    def __init__(self, h5_path, transform=None, verbose=True):
        self.h5_path = h5_path
        self.h5_file = None
        # self.length = len(h5py.File(h5_path, 'r'))
        if verbose:
            s0 = time.time()
            print(f'[H5Dataset] Loading {h5_path} to get records length... thread: {threading.get_native_id()}...')
        with h5py.File(h5_path, 'r') as f:
            self.length = len(f)
        if verbose:
            print(f'[H5Dataset] Loading {h5_path} to get records length... Done! Time: {time.time()-s0:.2f}s')
            print(f'[H5Dataset] Loaded {self.length} records...')
        self.transform = transform
        self.verbose = verbose

    def __getitem__(self, index):

        #loading in getitem allows us to use multiple processes for data loading
        #because hdf5 files aren't pickelable so can't transfer them across processes
        # https://discuss.pytorch.org/t/hdf5-a-data-format-for-pytorch/40379
        # https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
        # TODO possible look at __getstate__ and __setstate__ as a more elegant solution
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            if self.verbose:
                print(f'[H5Dataset] Instance gained, loading {self.h5_path}... Done! on thread: {threading.get_native_id()}...')

        record = self.h5_file[str(index)]

        if self.transform:
            x = Image.fromarray(record['data'][()])
            x = self.transform(x)
        else:
            x = torch.from_numpy(record['data'][()])

        y = record['target'][()]
        y = torch.from_numpy(np.asarray(y))

        return (x,y)

    def __len__(self):
        return self.length
