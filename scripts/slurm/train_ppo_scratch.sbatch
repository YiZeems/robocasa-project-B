#!/bin/bash
#SBATCH --job-name=robocasa_ppo_scratch
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/train_ppo_scratch_%j.out
#SBATCH --error=logs/slurm/train_ppo_scratch_%j.err

# Total steps can be overridden: sbatch train_ppo_scratch.sbatch 1000000
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-500000}
PYTHON=/home/infres/hfanchini-25/.conda/envs/robocasa_telecom/bin/python

mkdir -p logs/slurm

$PYTHON -m robocasa_telecom.rl.train \
    --config configs/train/open_single_door_ppo.yaml \
    --seed 0 \
    --total-timesteps $TOTAL_TIMESTEPS
