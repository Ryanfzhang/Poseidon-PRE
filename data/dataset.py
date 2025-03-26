import xarray as xr
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler

from typing import Tuple, List
import torch
import random
from torch.utils import data
from torchvision import transforms as T
import os

class NetCDFDataset(data.Dataset):
    """Dataset class for the era5 upper and surface variables."""

    def __init__(self,
                 dataset_path = '/home/mafzhang/data/cmoms/',
                 data_transform = None,
                 seed = 1234,
                 training = True,
                 validation = False,
                 startDate = '19880501',
                 endDate = '20191231',
                 freq = 'D',
                 lead_time = 7,
                 ):
        """Initialize."""
        self.dataset_path = dataset_path
        # Prepare the datetime objects for training, validation, and test
        self.training = training
        self.validation = validation
        self.data_transform = data_transform
        self.lead_time = lead_time
        self.mean = np.load(os.path.join(self.dataset_path, "mean.npy"))
        self.std = np.load(os.path.join(self.dataset_path, "std.npy"))
        self.mask = np.load(os.path.join(self.dataset_path, "mask.npy"))
        self.coastal = np.load(os.path.join(self.dataset_path, "coastal.npy"))
        self.weight = np.load(os.path.join(self.dataset_path, "weight.npy"))
        # self.scale = np.ones(19)
        # self.scale[11] = 0.01
        # self.scale[12] = 0.01
        # self.scale[13] = 0.1

        self.keys = list(pd.date_range(start=startDate, end=endDate, freq=freq))[1:]
        self.length = len(self.keys) - lead_time - 1

        random.seed(seed)

    def nctonumpy(self, dataset):
        """
        Input
            xr.Dataset
        Return
            numpy array 
        """
        data = dataset['data'].values.astype(np.float32) 
        return data

    def LoadData(self, key):
        """
        Input
            key: datetime object, input time
        Return
            input: numpy
            target: numpy label
            (start_time_str, end_time_str): string
        """
        # start_time datetime obj
        start_time = key
        start_time_str = datetime.strftime(key, '%Y%m%d')
        year, month, day = start_time_str[0:4], start_time_str[4:6], start_time_str[6:]
        input = np.load(os.path.join(self.dataset_path , "{}/{}-{}-{}.npy".format(year, year, month, day)))
        input_mark = np.stack([start_time.month - 1, start_time.day -1])
        input_climatology = np.load(os.path.join(self.dataset_path , "climatology/{}-{}.npy".format(month, day)))

        # start_time_minus_1 = key - timedelta(days=1)
        # start_time_minus_1_str = start_time_minus_1.strftime('%Y%m%d')
        # year, month, day = start_time_minus_1_str[0:4], start_time_minus_1_str[4:6], start_time_minus_1_str[6:]
        # input_minus_1 = np.load(os.path.join(self.dataset_path , "{}/{}-{}-{}.npy".format(year, year, month, day)))
        # input = np.stack([input, input_minus_1], axis=2)

        # input_minus_1_mark = np.stack([start_time_minus_1.month - 1, start_time_minus_1.day -1])
        # input_mark = np.stack([input_mark, input_minus_1_mark], axis=0)

        end_time = key + timedelta(days=self.lead_time)
        end_time_str = end_time.strftime('%Y%m%d')
        year, month, day = end_time_str[0:4], end_time_str[4:6], end_time_str[6:]
        target = np.load(os.path.join(self.dataset_path , "{}/{}-{}-{}.npy".format(year, year, month, day)))
        target_mark = np.stack([end_time.month - 1, end_time.day -1])
        target_climatology = np.load(os.path.join(self.dataset_path , "climatology/{}-{}.npy".format(month, day)))

        return input, input_mark, target, target_mark, (start_time_str, input_climatology, end_time_str, target_climatology)

    def __getitem__(self, index):
        """Return input frames, target frames, and its corresponding time steps."""
        iii = self.keys[index]
        input, input_mark, target, target_mark, periods = self.LoadData(iii)
        input = self.normalize(input)
        target = self.normalize(target)

        input = np.nan_to_num(input, nan=0.)
        target = np.nan_to_num(target, nan=0.)
        info={}
        info['start_time'] = periods[0]
        info['end_time'] = periods[2]
        info['mean'] = self.mean
        info['std'] = self.std
        info['mask'] = self.mask[0, 0]
        info['coastal'] = self.coastal
        info['weight'] = self.weight
        info['input_climatology'] = periods[1]
        info['target_climatology'] = periods[3]

        return input, input_mark.astype(np.float32), target, target_mark.astype(np.float32), info
    
    def normalize(self, input):
        output = (input - self.mean[:, :, np.newaxis, np.newaxis])/(self.std[:, :, np.newaxis, np.newaxis])
        return output

    def denormalize(self, input):
        output = input * self.mean[:, :, np.newaxis, np.newaxis, np.newaxis] + self.mean[:, :, np.newaxis, np.newaxis, np.newaxis]
        return output

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__


if __name__=="__main__":
    dataset = NetCDFDataset()
    print(dataset.__getitem__(dataset.__len__()-1)[0].shape)
