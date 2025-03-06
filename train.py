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
            batch_mask = batch_mask.transpose(1,2)
            loss = (loss* batch_mask).mean()
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        train_loss.update(loss.item())
    accelerator.print("Epoch: {}| Train Loss: {:.4f}, Cost Time: {:.4f}".format(epoch, train_loss.avg, time.time()-epoch_time))
    # if epoch%30==0 and epoch!=0:
    #     with torch.no_grad():
    #         mean_mae_sst, mean_mse_sst, mean_rmse_sst, mean_mape_sst, mean_mspe_sst =AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    #         mean_mae_salt, mean_mse_salt, mean_rmse_salt, mean_mape_salt, mean_mspe_salt = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    #         for i, (ninput, _, input_mark, noutput, _, output_mark, _) in enumerate(test_dloader):
    #             ninput = ninput[:,-2:,:,:]
    #             noutput = noutput[:,-2:,:,:]
    #             ninput, input_mark, noutput, output_mark = ninput.float().squeeze().to(device), input_mark.int().squeeze().to(device), noutput.float().squeeze().to(device), output_mark.int().squeeze().to(device)
    #             latents = torch.randn(
    #                 noutput.shape,
    #                 generator=generator,
    #             ).to(device)
    #             for t in noise_scheduler.timesteps:
    #                 timesteps=torch.ones(ninput.shape[0]).long().to(device)
    #                 with torch.no_grad():
    #                     noise_pred = model(ninput, input_mark, latents, output_mark, t*timesteps)
    #                 latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

    #             pred = accelerator.gather(latents + ninput) 
    #             truth = accelerator.gather(noutput)
    #             pred = pred.detach().cpu().numpy()
    #             truth = truth.detach().cpu().numpy()
    #             # pred = pred * std + mean
    #             # truth = truth * std + mean
    #             pred_sst = pred[:,0,:,:]
    #             truth_sst = truth[:,0,:,:]
    #             pred_salt = pred[:,1,:,:]
    #             truth_salt = truth[:,1,:,:]

    #             mae_sst, mse_sst, rmse_sst, mape_sst, mspe_sst = metric(pred_sst, truth_sst)
    #             mae_salt, mse_salt, rmse_salt, mape_salt, mspe_salt = metric(pred_salt, truth_salt)
    #             mean_mae_sst.update(mae_sst)
    #             mean_mse_sst.update(mse_sst)
    #             mean_rmse_sst.update(rmse_sst)
    #             mean_mape_sst.update(mape_sst)
    #             mean_mspe_sst.update(mspe_sst)
    #             mean_mae_salt.update(mae_salt)
    #             mean_mse_salt.update(mse_salt)
    #             mean_rmse_salt.update(rmse_salt)
    #             mean_mape_salt.update(mape_salt)
    #             mean_mspe_salt.update(mspe_salt)

    #         accelerator.print("*"*100)
    #         accelerator.print("{}-th Epoch Test-> SST MSE : {:.6f}, MAE : {:.6f}, RMSE : {:.6f}, MAPE : {:.6f}, MSPE : {:.6f}\n".format(epoch, mean_mse_sst.avg, mean_mae_sst.avg, mean_rmse_sst.avg, mean_mape_sst.avg, mean_mspe_sst.avg))
    #         accelerator.print("{}-th Epoch Test-> Salt MSE : {:.6f}, MAE : {:.6f}, RMSE : {:.6f}, MAPE : {:.6f}, MSPE : {:.6f}\n".format(epoch, mean_mse_salt.avg, mean_mae_salt.avg, mean_rmse_salt.avg, mean_mape_salt.avg, mean_mspe_salt.avg))
    #     if mean_mse_sst.avg < best_mse_sst and mean_mse_salt.avg < best_mse_salt:
    #         if accelerator.is_main_process:
    #             best_mse_sst = mean_mse_sst.avg
    #             best_mse_salt = mean_mse_salt.avg
    #             torch.save(model.state_dict(), os.path.join(args.checkpoints, 'model_best.pth'))
    #             accelerator.print("Best Model Saved")