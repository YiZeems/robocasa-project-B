#!/bin/bash
#SBATCH --job-name=robocasa_debug
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/debug_reward_%j.out
#SBATCH --error=logs/slurm/debug_reward_%j.err

PYTHON=/home/infres/hfanchini-25/.conda/envs/robocasa_telecom/bin/python
CHECKPOINT=${1:-""}   # optional: pass checkpoint path as $1

mkdir -p logs/slurm

if [ -z "$CHECKPOINT" ]; then
    $PYTHON -m robocasa_telecom.tools.debug_reward \
        --config configs/train/open_single_door_ppo.yaml \
        --random \
        --episodes 3 \
        --seed 0
else
    $PYTHON -m robocasa_telecom.tools.debug_reward \
        --config configs/train/open_single_door_ppo.yaml \
        --checkpoint "$CHECKPOINT" \
        --episodes 3 \
        --seed 0
fi
