#! /bin/bash
#SBATCH --job-name=DiT
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --time=01:20:00
#SBATCH -p earthhk

###############################

accelerate launch --use_deepspeed train.py --dataset_path="/project/earthhk/cmoms/"

###############################