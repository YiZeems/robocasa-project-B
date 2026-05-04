# Runbook

## Pré-requis

- Linux recommandé pour cluster, macOS/Linux possibles en local
- `uv` installé et disponible dans le `PATH`
- `git` installé
- accès réseau au moment du setup

## Setup

```bash
bash scripts/setup_uv.sh
```

Exemple complet:

```bash
PYTHON_VERSION=3.11 \
ROBOCASA_COMMIT=9a3a78680443734786c9784ab661413edb87067b \
ROBOSUITE_COMMIT=aaa8b9b214ce8e77e82926d677b4d61d55e577ab \
DOWNLOAD_ASSETS=1 \
VERIFY_ASSETS=1 \
DOWNLOAD_DATASETS=0 \
RUN_SETUP_MACROS=1 \
bash scripts/setup_uv.sh
```

## Validation

```bash
uv run python -m robocasa_telecom.sanity --config configs/env/open_single_door.yaml --steps 20
```

## Train

```bash
uv run python -m robocasa_telecom.train --config configs/train/open_single_door_ppo.yaml --seed 0
```

Ou:

```bash
bash scripts/run_train.sh configs/train/open_single_door_ppo.yaml 0
```

Reprise d'un run interrompu:

```bash
uv run python -m robocasa_telecom.train \
  --config configs/train/open_single_door_sac.yaml \
  --seed 0 \
  --resume-from checkpoints/<run_id>/sac_100000_steps.zip
```

Le wrapper `scripts/run_train.sh` accepte aussi un 3e argument `RESUME_FROM`.

## Eval

```bash
uv run python -m robocasa_telecom.evaluate \
  --config configs/train/open_single_door_ppo.yaml \
  --checkpoint checkpoints/<run_id>/final_model.zip \
  --num-episodes 20 \
  --deterministic
```

## SLURM

```bash
sbatch scripts/slurm/train_array.sbatch
sbatch --export=ALL,CHECKPOINT_PATH=checkpoints/<run_id>/final_model.zip scripts/slurm/eval.sbatch
```

## Dépannage

- Si `uv run` échoue, vérifiez que `.venv` a bien été créé par `scripts/setup_uv.sh`.
- Si des assets manquent, relancez avec `DOWNLOAD_ASSETS=1 VERIFY_ASSETS=1`.
- Sur Windows, privilégiez WSL ou Git Bash.
