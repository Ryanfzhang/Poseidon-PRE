#! /bin/bash
#SBATCH --job-name=DiT
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=72:00:00
#SBATCH -p normal
#SBATCH --mail-user=mafzhang@ust.hk
#SBATCH --account=earthhk
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err


###############################

accelerate launch train.py --dataset_path="/project/earthhk/cmoms/"

###############################