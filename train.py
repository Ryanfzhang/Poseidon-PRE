import argparse
import os
import time
import torch
import random
import numpy as np
from accelerate import Accelerator
from tqdm import tqdm
from timm.utils import AverageMeter
from timm.utils import AverageMeter
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
import pandas as pd

from utils import check_dir, seed_everything
from data.dataset import NetCDFDataset
from backbones.model import OceanTransformer

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
fix_seed = 2025
seed_everything(fix_seed)

parser = argparse.ArgumentParser(description='OceanFormer Forecasting')

# data loader
parser.add_argument('--freq', type=str, default='d',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
# forecasting task
parser.add_argument('--lead_time', type=int, default=7, help='input sequence length')

# optimization
parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')
parser.add_argument('--batch_size', type=int, default=3, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=1e-6, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='mae', help='loss function')

args = parser.parse_args()

check_dir(args.checkpoints)
accelerator = Accelerator(gradient_accumulation_steps=1)
device = accelerator.device

train_dataset = NetCDFDataset(lead_time=args.lead_time)
train_dloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, prefetch_factor=4, num_workers=4)
test_dataset = NetCDFDataset(startDate='20200101', endDate='20221228')
test_dloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, prefetch_factor=4, num_workers=4)

model = OceanTransformer()
optimizer = torch.optim.Adamax(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps= 1000, 
    num_training_steps=len(train_dloader) * args.train_epochs,
)

train_dloader, test_dloader, model, optimizer, lr_scheduler = accelerator.prepare(train_dloader, test_dloader, model, optimizer, lr_scheduler)
mask = torch.from_numpy(train_dataset.mask).to(device)
mean = torch.from_numpy(train_dataset.mean).to(device)
std = torch.from_numpy(train_dataset.std).to(device)
criteria = torch.nn.L1Loss(reduce=False)

best_mse_sst, best_mse_salt = 100, 100
for epoch in tqdm(range(args.train_epochs)):
    train_loss = AverageMeter()
    model.train()
    epoch_time = time.time()
    for i, (input, input_mark, output, output_mark, _) in tqdm(enumerate(train_dloader), total=len(train_dloader), disable=not accelerator.is_local_main_process):
        input, input_mark, output, output_mark = input.float().to(device), input_mark.int().to(device), output.float().to(device), output_mark.int().to(device)
        input = input.transpose(1,2)
        output = output.transpose(1,2)

        optimizer.zero_grad()
        with accelerator.accumulate(model):
            pred = model(input)
            loss = criteria(pred, output)
            batch_mask = mask.unsqueeze(0).expand(pred.shape[0], -1, -1, -1, -1)
            batch_mask = 1. - batch_mask.transpose(1,2)
            loss = (loss* batch_mask).mean()
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        train_loss.update(loss.item())
    accelerator.print("Epoch: {}| Train Loss: {:.4f}, Cost Time: {:.4f}".format(epoch, train_loss.avg, time.time()-epoch_time))
    if epoch%10==0:
        with torch.no_grad():
            rmse_list, mape_list =[], []
            for i, (input, input_mark, output, output_mark, _) in tqdm(enumerate(), total=len(train_dloader), disable=not accelerator.is_local_main_process):
                input, input_mark, output, output_mark = input.float().to(device), input_mark.int().to(device), output.float().to(device), output_mark.int().to(device)
                input = input.transpose(1,2)
                output = output.transpose(1,2)

                pred = model(input)

                batch_mean = mean.unsqueeze(0).expand(pred.shape[0], -1, -1, -1, -1)
                batch_std = std.unsqueeze(0).expand(pred.shape[0], -1, -1, -1, -1)
                batch_mask = mask.unsqueeze(0).expand(pred.shape[0], -1, -1, -1, -1)
                batch_mask = 1. - batch_mask.transpose(1,2)

                pred = accelerator.gather(pred * batch_std + batch_mean) 
                truth = accelerator.gather(output * batch_std + batch_mean)
                batch_mask = accelerator.gather(batch_mask)
                pred = pred.detach().cpu().numpy()
                truth = truth.detach().cpu().numpy()
                batch_mask = batch_mask.detach().cpu().numpy()

                rmse = np.mean(np.sqrt(np.sum((pred - truth)**2 * batch_mask, axis=(2,3))/(np.sum(batch_mask, axis=(2,3)) + 1e-10)), axis=0)
                mape = np.mean(np.sum(np.abs((pred - truth) / truth) * batch_mask, axis=(2,3))/(np.sum(batch_mask, axis=(2,3)) + 1e-10), axis=0)
                rmse_list.append(rmse)
                mape.append(mape)

            all_rmse = np.stack(rmse_list, axis=0)
            all_mape = np.stack(mape_list, axis=0)
            mean_rmse = np.mean(all_rmse, axis=0)
            mean_mape = np.mean(all_mape, axis=0)

            accelerator.print("*"*100)
            rmse = pd.DataFrame(mean_rmse)
            mape = pd.DataFrame(mean_mape)
            print("RMSE for all level and all drivers:\n")
            print(rmse)
            print("MAPE for all level and all drivers:\n")
            print(mape)

        if accelerator.is_main_process:
            torch.save(model.state_dict(), os.path.join(args.checkpoints, 'model_best.pth'))