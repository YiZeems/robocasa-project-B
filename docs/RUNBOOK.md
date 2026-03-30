# Runbook (Local + SLURM)

## Local

1. Setup env:

```bash
bash scripts/setup_conda.sh
conda activate robocasa_telecom
```

2. Sanity:

```bash
python -m robocasa_telecom.sanity --config configs/env/open_single_door.yaml --steps 20
```

3. Train court:

```bash
python -m robocasa_telecom.train --config configs/train/open_single_door_ppo.yaml --seed 0 --total-timesteps 5000
```

4. Eval:

```bash
python -m robocasa_telecom.evaluate --config configs/train/open_single_door_ppo.yaml --checkpoint checkpoints/<run_id>/final_model.zip --num-episodes 20 --deterministic
```

## SLURM

## Train array

```bash
sbatch scripts/slurm/train_array.sbatch
```

Variables utiles:
- `SEEDS_CSV=0,1,2`
- `CONFIG_PATH=configs/train/open_single_door_ppo.yaml`
- `ENV_NAME=robocasa_telecom`

## Eval job

```bash
sbatch --export=ALL,CHECKPOINT_PATH=checkpoints/<run_id>/final_model.zip scripts/slurm/eval.sbatch
```

## Erreurs fréquentes

## `FileNotFoundError` assets RoboCasa

Cause: assets non téléchargés.

Fix:
```bash
DOWNLOAD_ASSETS=1 bash scripts/setup_conda.sh
```
ou
```bash
python -m robocasa.scripts.download_kitchen_assets --type all
```

## Env task introuvable

Le code mappe automatiquement:
- `OpenSingleDoor` -> `OpenCabinet`
- `OpenDoor` -> `OpenCabinet`
