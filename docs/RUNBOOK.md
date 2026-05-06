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

Ce setup :
- clone `external/robosuite` et `external/robocasa`,
- installe le projet avec `uv sync`,
- installe `robocasa` et `robosuite` depuis les sources locales déclarées dans `pyproject.toml`,
- relie automatiquement `robocasa/models/assets` vers les assets du checkout externe si besoin,
- vérifie les imports et la présence d'assets critiques.

## Validation

```bash
make check
uv run python -m robocasa_telecom.sanity --config configs/env/open_single_door.yaml --steps 20
```

Smoke train recommandé :

```bash
uv run python -m robocasa_telecom.train \
  --config configs/train/open_single_door_sac_debug.yaml \
  --seed 0 \
  --total-timesteps 10 \
  --no-auto-resume
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
Il est conservé pour le confort, mais la commande standard reste `uv run`.

Sans `--resume-from`, l’entraînement active `--auto-resume` par défaut et
reprend le dernier run interrompu correspondant au même `task/algo/seed`.
Utilise `--no-auto-resume` pour forcer un départ depuis zéro.

Pour générer automatiquement une vidéo 4 vues du meilleur checkpoint à la fin
du `train`, activer:

```bash
ROBOCASA_RENDER_BEST_RUN_VIDEO=1
```

La vidéo est alors écrite sous `outputs/<run_id>/videos/` avec un fichier JSON
de métadonnées adjacent. Pour désactiver le hook, mettre `ROBOCASA_RENDER_BEST_RUN_VIDEO=0`.

Optionnellement, on peut raccourcir le smoke test ou ajuster le rendu via:

```bash
ROBOCASA_RENDER_BEST_RUN_VIDEO_MAX_STEPS=20
ROBOCASA_RENDER_BEST_RUN_VIDEO_FPS=20
ROBOCASA_RENDER_BEST_RUN_VIDEO_MIN_SECONDS=12
ROBOCASA_RENDER_BEST_RUN_VIDEO_MAX_EPISODES=5
```

Commandes utiles :

```bash
uv run python -m robocasa_telecom.train --config configs/train/open_single_door_sac_debug.yaml --seed 0
uv run python -m robocasa_telecom.train --config configs/train/open_single_door_sac.yaml --seed 0
uv run python -m robocasa_telecom.train --config configs/train/open_single_door_sac_tuned.yaml --seed 0
uv run python -m robocasa_telecom.train --config configs/train/open_single_door_ppo.yaml --seed 0
uv run python -m robocasa_telecom.train --config configs/train/open_single_door_ppo_baseline.yaml --seed 0
```

## Eval

```bash
uv run python -m robocasa_telecom.evaluate \
  --config configs/train/open_single_door_ppo.yaml \
  --checkpoint checkpoints/<run_id>/final_model.zip \
  --num-episodes 20 \
  --deterministic
```

Le wrapper `scripts/run_eval.sh` existe aussi pour les usages rapides, mais
`uv run` reste la voie recommandée.

## SLURM

```bash
sbatch scripts/slurm/train_array.sbatch
sbatch --export=ALL,CHECKPOINT_PATH=checkpoints/<run_id>/final_model.zip scripts/slurm/eval.sbatch
sbatch --export=ALL,CONFIG_PATH=configs/train/open_single_door_sac.yaml,CHECKPOINT_PATH=checkpoints/<run_id>/best_model.zip,SEED=0 scripts/slurm/render_best_run.sbatch
```

## Dépannage

- Si `uv run` échoue, vérifiez que `.venv` a bien été créé par `scripts/setup_uv.sh`.
- Si des assets manquent, relancez `bash scripts/setup_uv.sh` avec `DOWNLOAD_ASSETS=1 VERIFY_ASSETS=1`.
- Si le `GymWrapper` RoboSuite donne une shape d'observation instable, le projet bascule automatiquement sur `RawRoboCasaAdapter`. Ce fallback est attendu.
- Sur Windows, privilégiez WSL ou Git Bash.
