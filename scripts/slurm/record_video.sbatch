#!/bin/bash
#SBATCH --job-name=robocasa_video
#SBATCH --partition=P100
#SBATCH --output=logs/slurm/record_video_%j.out
#SBATCH --error=logs/slurm/record_video_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

cd ~/projects/Robocasa
mkdir -p logs/slurm

module purge
module load miniconda3/25.5.1
module load cuda/11.8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate robocasa_telecom

CHECKPOINT="${CHECKPOINT:-checkpoints/OpenCabinet_ppo_seed0_resume_501k_to_1001k_20260503_185022/final_model.zip}"
OUTPUT="${OUTPUT:-eval_video_${SLURM_JOB_ID}.mp4}"
EPISODES="${EPISODES:-5}"

echo "Recording video for checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT"

python -m robocasa_telecom.record_video \
  --config configs/train/open_single_door_ppo.yaml \
  --checkpoint "$CHECKPOINT" \
  --episodes "$EPISODES" \
  --output "$OUTPUT" \
  --width 256 \
  --height 256

echo "Done."
