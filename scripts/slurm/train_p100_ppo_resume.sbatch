#!/bin/bash
#SBATCH --job-name=ppo_resume
#SBATCH --partition=P100
#SBATCH --output=logs/slurm/resume_ppo_%j.out
#SBATCH --error=logs/slurm/resume_ppo_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

cd ~/projects/Robocasa

mkdir -p logs/slurm

module purge
module load miniconda3/25.5.1
module load cuda/11.8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate robocasa_telecom

echo "PPO resume started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
date
nvidia-smi

python -m robocasa_telecom.train \
  --config configs/train/open_single_door_ppo.yaml \
  --seed 0 \
  --total-timesteps 500000 \
  --resume-from checkpoints/OpenCabinet_ppo_seed0_20260502_124604/final_model.zip

date
echo "PPO resume finished"
