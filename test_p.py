import argparse
import os
import time
import torch
import random
import numpy as np
from accelerate import Accelerator
from tqdm import tqdm
from timm.utils import AverageMeter
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
import pandas as pd

from utils import check_dir, seed_everything
from data.dataset import NetCDFDataset
from model.oceanformer import Xuanming

# os.environ['CUDA_LAUNCH_BLOCKING']="1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"
fix_seed = 2025
seed_everything(fix_seed)

parser = argparse.ArgumentParser(description='OceanFormer Forecasting')

# data loader
parser.add_argument('--freq', type=str, default='d',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--dataset_path', type=str, default='/home/mafzhang/data/cmoms/', help='location of dataset')

# forecasting task
parser.add_argument('--lead_time', type=int, default=7, help='input sequence length')
parser.add_argument('--levels', type=int, default=30, help='input sequence length')
parser.add_argument('--drivers', type=int, default=19, help='input sequence length')
parser.add_argument('--depth', type=int, default=24, help='input sequence length')
parser.add_argument('--hidden_size', type=int, default=1024, help='input sequence length')

# optimization
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=2, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='optimizer learning rate')
parser.add_argument('--weight_decay', type=float, default=3e-6, help='optimizer wd')
parser.add_argument('--loss', type=str, default='mae', help='loss function')

args = parser.parse_args()

check_dir(args.checkpoints)
accelerator = Accelerator()

test_dataset = NetCDFDataset(startDate='20200101', endDate='20221228', lead_time=args.lead_time, dataset_path=args.dataset_path)
test_dloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=8, prefetch_factor=4)

test_dloader = accelerator.prepare_data_loader(test_dloader)
criteria = torch.nn.L1Loss(reduction='none')

best_mse_sst, best_mse_salt = 100, 100
with torch.no_grad():
    all_rmse = 0
    for i, (input, input_mark, output, output_mark, info) in tqdm(enumerate(test_dloader), total=len(test_dloader), disable=(not accelerator.is_local_main_process)):
        input = input.transpose(1,2)
        output = output.transpose(1,2)

        mean = info['mean'].unsqueeze(-1).unsqueeze(-1)
        std = info['std'].unsqueeze(-1).unsqueeze(-1)
        mean = mean.transpose(1,2)
        std = std.transpose(1,2)
        mask = 1. - info['mask'].unsqueeze(1).unsqueeze(1)

        pred = input * std + mean
        truth = output * std + mean
        rmse = torch.mean((pred - truth)**2 * mask, dim=0, keepdim=True)
        rmse = accelerator.gather(rmse)
        rmse = rmse.detach().cpu().numpy()
        torch.cuda.empty_cache()

        all_rmse = all_rmse + rmse

    if accelerator.is_main_process:
        all_rmse = all_rmse / len(test_dloader)
        mean_rmse = np.sqrt(np.mean(all_rmse, axis=0))
        np.save(os.path.join(args.checkpoints, 'horizontal_rmse_p.npy'), mean_rmse)
        print(mean_rmse)
        print(mean_rmse.shape)