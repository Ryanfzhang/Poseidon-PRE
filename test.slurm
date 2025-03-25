#! /bin/bash
#SBATCH --job-name=OceanForecaster
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH -p normal
#SBATCH --mail-user=mafzhang@ust.hk
#SBATCH --time=72:00:00
#SBATCH --account=earthhk
#SBATCH --output=logs/%j.out
#SBATCH --cpus-per-gpu=8

###############################

accelerate launch test.py --dataset_path="/project/earthhk/cmoms/"

###############################