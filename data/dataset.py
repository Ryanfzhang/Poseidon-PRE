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
                 dataset_path = '/home/mafzhang/code/Project/ocean-fundation-model-pre/dataset/cmoms/',
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

        self.keys = list(pd.date_range(start=startDate, end=endDate, freq=freq))
        self.length = len(self.keys) - lead_time

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

        # convert datetime obj to string for matching file name and return key
        start_time_str = datetime.strftime(key, '%Y%m%d')
        end_time = key + timedelta(days=self.lead_time)
        end_time_str = end_time.strftime('%Y%m%d')
        
        # Prepare the input_surface dataset
        year, month, day = start_time_str[0:4], start_time_str[4:6], start_time_str[6:]
        input = np.load(os.path.join(self.dataset_path , "{}/{}-{}-{}.npy".format(year, year, month, day)))
        input_mark = np.stack([start_time.month - 1, start_time.day -1])

        year, month, day = end_time_str[0:4], end_time_str[4:6], end_time_str[6:]
        target = np.load(os.path.join(self.dataset_path , "{}/{}-{}-{}.npy".format(year, year, month, day)))
        target_mark = np.stack([end_time.month - 1, end_time.day -1])

        return input, input_mark, target, target_mark, (start_time_str, end_time_str)

    def __getitem__(self, index):
        """Return input frames, target frames, and its corresponding time steps."""
        iii = self.keys[index]
        input, input_mark, target, target_mark, periods = self.LoadData(iii)
        input = self.normalize(input)
        target = self.normalize(target)
        input = np.nan_to_num(input, nan=0.)
        target = np.nan_to_num(target, nan=0.)

        return input, input_mark, target, target_mark, periods
    
    def normalize(self, input):
        output = (input- self.mean[:, :, np.newaxis, np.newaxis])/(self.std[:, :, np.newaxis, np.newaxis])
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
    print(dataset.__getitem__(dataset.__len__()-1))
