#! /bin/bash
#SBATCH --job-name=Poseidon_recon
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH -p normal
#SBATCH --mail-user=mafzhang@ust.hk
#SBATCH --time=72:00:00
#SBATCH --account=earthhk
#SBATCH --output=logs/poseidon_pre.out

###############################

accelerate launch train_pre.py --dataset_path="/project/earthhk/cmoms/" --depth 12 --weight_decay 5e-5 --patch_size 8

###############################
