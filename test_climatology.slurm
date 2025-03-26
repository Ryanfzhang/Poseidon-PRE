#! /bin/bash
#SBATCH --job-name=evaluate_oceanforecaster
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH -p normal
#SBATCH --mail-user=mafzhang@ust.hk
#SBATCH --time=2:00:00
#SBATCH --account=earthhk
#SBATCH --output=logs/evaluate_%j.out
#SBATCH --cpus-per-gpu=8

###############################

accelerate launch test_climatology.py --dataset_path="/project/earthhk/cmoms/"

###############################