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
from backbones.model import OceanTransformer

# os.environ['CUDA_LAUNCH_BLOCKING']="1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"
fix_seed = 2025
seed_everything(fix_seed)

parser = argparse.ArgumentParser(description='OceanFormer Forecasting')

# data loader
parser.add_argument('--freq', type=str, default='d',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
# forecasting task
parser.add_argument('--lead_time', type=int, default=7, help='input sequence length')
parser.add_argument('--levels', type=int, default=30, help='input sequence length')
parser.add_argument('--drivers', type=int, default=19, help='input sequence length')

# optimization
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=4, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='optimizer learning rate')
parser.add_argument('--weight_decay', type=float, default=3e-6, help='optimizer wd')
parser.add_argument('--loss', type=str, default='mae', help='loss function')

args = parser.parse_args()

check_dir(args.checkpoints)
accelerator = Accelerator()
device = accelerator.device

test_dataset = NetCDFDataset(startDate='20200101', endDate='20221228', lead_time=args.lead_time)
test_dloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=True)

model = OceanTransformer()
params = torch.load(os.path.join(args.checkpoints, 'model_best.pth'))
from collections import OrderedDict
new_params = OrderedDict()
for k, v in params.items():
    name = k[7:] 
    new_params[name] = v

model.load_state_dict(new_params)
del new_params, params

test_dloader, model = accelerator.prepare(test_dloader, model)
mask = torch.from_numpy(test_dataset.mask).to(device)
mean = torch.from_numpy(test_dataset.mean).to(device)
std = torch.from_numpy(test_dataset.std).to(device)
criteria = torch.nn.L1Loss(reduction='none')

best_mse_sst, best_mse_salt = 100, 100
with torch.no_grad():
    all_rmse = 0
    count = 0
    for i, (input, input_mark, output, output_mark, _) in tqdm(enumerate(test_dloader), total=len(test_dloader), disable=not accelerator.is_local_main_process):
        input, input_mark, output, output_mark = input.float().to(device), input_mark.int().to(device), output.float().to(device), output_mark.int().to(device)
        input = input.transpose(1,2)
        output = output.transpose(1,2)

        pred = model(input)

        batch_mean = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(pred.shape[0], -1, -1, -1, -1)
        batch_std= std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(pred.shape[0], -1, -1, -1, -1)
        batch_mean = batch_mean.transpose(1,2)
        batch_std = batch_std.transpose(1,2)
        batch_mask = mask.unsqueeze(0).expand(pred.shape[0], -1, -1, -1, -1)
        batch_mask = 1. - batch_mask.transpose(1,2)

        pred = pred * batch_std + batch_mean
        truth = output * batch_std + batch_mean
        rmse = torch.mean((pred - truth)**2 * batch_mask, dim=0, keepdim=True)
        rmse = accelerator.gather(rmse)
        rmse = rmse.detach().cpu().numpy()
        torch.cuda.empty_cache()

        all_rmse = all_rmse + rmse
        count += 1

    if accelerator.is_main_process:
        all_rmse = all_rmse / count
        mean_rmse = np.sqrt(np.mean(all_rmse, axis=0))
        print(mean_rmse)
        print(mean_rmse.shape)