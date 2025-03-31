#! /bin/bash
#SBATCH --job-name=OceanForecaster
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH -p normal
#SBATCH --mail-user=mafzhang@ust.hk
#SBATCH --time=72:00:00
#SBATCH --account=earthhk
#SBATCH --output=logs/oceanformer_d12_wd5e-5.out
#SBATCH --cpus-per-gpu=8

###############################

accelerate launch train.py --dataset_path="/project/earthhk/cmoms/" --depth 12 --weight_decay 5e-5

###############################