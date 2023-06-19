#!/bin/bash

#SBATCH --job-name=ara-poet
#SBATCH --error=logs/%j%x.err # error file
#SBATCH --output=logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
##SBATCH --wait-all-nodes=1


# srun python v2.py

srun python train.py config/finetune_arabic.py
