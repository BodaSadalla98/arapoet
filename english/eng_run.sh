#!/bin/bash

#SBATCH --job-name=eng-shakespeare
#SBATCH --error=eng_logs/%j%x.err # error file
#SBATCH --output=eng_logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
##SBATCH --wait-all-nodes=1


# srun python v2.py

srun python run_clm.py eng_config.json

