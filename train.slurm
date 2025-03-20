#! /bin/bash
#SBATCH --job-name=DiT
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH -p normal
#SBATCH --mail-user=mafzhang@ust.hk
#SBATCH --account=earthhk
#SBATCH --output=logs/slurm_%j.out

###############################

accelerate launch train.py --dataset_path="/project/earthhk/cmoms/"

###############################