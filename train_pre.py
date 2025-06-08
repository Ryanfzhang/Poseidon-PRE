import argparse
import os
import time
import torch
import random
import numpy as np
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm import tqdm
from timm.utils import AverageMeter
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
import pandas as pd

from utils import check_dir, seed_everything
from data.dataset import NetCDFDataset
from poseidon.poseidon import poseidon_recon, poseidon_pre

# os.environ['CUDA_LAUNCH_BLOCKING']="1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"
fix_seed = 2025
seed_everything(fix_seed)

parser = argparse.ArgumentParser(description='Ocean Forecasting')

# data loader
parser.add_argument('--freq', type=str, default='d',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/poseidon/', help='location of model checkpoints')
parser.add_argument('--dataset_path', type=str, default='/home/mafzhang/data/cmoms/', help='location of dataset')

# forecasting task
parser.add_argument('--lead_time', type=int, default=7, help='input sequence length')
parser.add_argument('--levels', type=int, default=30, help='input sequence length')
parser.add_argument('--patch_size', type=int, default=4, help='input sequence length')
parser.add_argument('--drivers', type=int, default=19, help='input sequence length')

# model
parser.add_argument('--depth', type=int, default=24, help='input sequence length')
parser.add_argument('--hidden_size', type=int, default=1024, help='input sequence length')
parser.add_argument('--beta', type=float, default=0.2, help='input sequence length')

# optimization
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=2, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='optimizer learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='optimizer wd')
parser.add_argument('--loss', type=str, default='mae', help='loss function')

args = parser.parse_args()


train_dataset = NetCDFDataset(dataset_path=args.dataset_path, lead_time=args.lead_time)
train_dloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, prefetch_factor=4)
test_dataset = NetCDFDataset(startDate='20200101', endDate='20221228', dataset_path=args.dataset_path, lead_time=args.lead_time)
test_dloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=8, prefetch_factor=4)

model = poseidon_pre(patch_size=args.patch_size, depth=args.depth)
model.load_state_dict(torch.load(os.path.join(args.checkpoints, "model_best.pth")), strict=False)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.995))
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps= 1000, 
    num_training_steps=len(train_dloader) * args.train_epochs,
)

accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
device = accelerator.device
train_dloader = accelerator.prepare_data_loader(train_dloader)
test_dloader = accelerator.prepare_data_loader(test_dloader)
model = accelerator.prepare_model(model)
optimizer = accelerator.prepare_optimizer(optimizer)
lr_scheduler = accelerator.prepare_scheduler(lr_scheduler)

criteria = torch.nn.L1Loss(reduction='none')

best_mse_sst, best_mse_salt = 100, 100

if accelerator.is_main_process:
    check_dir(args.checkpoints)
accelerator.print("Training start")

for epoch in range(args.train_epochs):
    train_loss = AverageMeter()
    model.train()
    epoch_time = time.time()
    # for i, (input, input_mark, output, output_mark, info) in enumerate(train_dloader):
    for i, (input, input_mark, output, output_mark, info) in tqdm(enumerate(train_dloader), total=len(train_dloader), disable=(not accelerator.is_local_main_process)):
        input = input.transpose(1,2)
        output = output.transpose(1,2)

        optimizer.zero_grad()
        pred = model(input, input_mark, output_mark)
        loss = criteria(pred, output)
        mask = 1. - info['mask'].unsqueeze(1).unsqueeze(1)
        coastal = info['coastal'].unsqueeze(1).unsqueeze(1)
        weight = info['weight'].unsqueeze(-1).unsqueeze(-1)
        coastal = coastal + (1. - coastal) * args.beta

        # loss = ((weight * loss) * mask).mean()
        loss = ((weight * (coastal * loss)) * mask).mean()
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        lr_scheduler.step()
        train_loss.update(loss.detach().cpu().item())
        torch.cuda.empty_cache()

    accelerator.print("Epoch: {}| Train Loss: {:.4f}, Cost Time: {:.4f}".format(epoch, train_loss.avg, time.time()-epoch_time))
    train_loss.reset()

    if epoch%10==0: 
        with torch.no_grad():
            model.eval()
            rmse_list =[]
            for i, (input, input_mark, output, output_mark, info) in enumerate(test_dloader):
                input = input.transpose(1,2)
                output = output.transpose(1,2)
                pred = model(input, input_mark, output_mark)

                mean = info['mean'].unsqueeze(-1).unsqueeze(-1)
                std = info['std'].unsqueeze(-1).unsqueeze(-1)
                mean = mean.transpose(1,2)
                std = std.transpose(1,2)
                mask = 1. - info['mask'].unsqueeze(1).unsqueeze(1)

                pred = pred * std + mean
                truth = output * std + mean

                rmse = torch.mean(torch.sqrt(torch.sum(torch.sum((pred - truth)**2 * mask, -1), dim=-1)/(torch.sum(torch.sum(mask, dim=-1), dim=-1) + 1e-10)), dim=0)
                rmse = accelerator.gather(rmse)
                rmse = rmse.detach().cpu().numpy()
                torch.cuda.empty_cache()

                rmse_list.append(rmse.reshape(-1, args.drivers, args.levels))

            all_rmse = np.concatenate(rmse_list, axis=0)
            mean_rmse = np.mean(all_rmse, axis=0)

            accelerator.print("*"*100)
            rmse = pd.DataFrame(mean_rmse)

            if accelerator.is_main_process:
                print("RMSE of all level and all drivers for reconstruct:\n")
                print(rmse)
                torch.save(model.state_dict(), os.path.join(args.checkpoints, 'model_best_pre.pth'))
                rmse.to_csv(os.path.join(args.checkpoints, 'rmse_pre_{}.csv'.format(epoch)))
