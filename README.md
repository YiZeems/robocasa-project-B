# RoboCasa Project B (Telecom)

Squelette exécutable pour entraîner et évaluer un agent RL sur RoboCasa, avec un flux d'installation basé sur `uv`.

## Objectif

- entraîner/évaluer une baseline PPO sur la tâche `OpenCabinet` (alias gérés: `OpenSingleDoor`, `OpenDoor`);
- exécuter en local et sur cluster SLURM GPU;
- produire des artefacts reproductibles.

## Compatibilité OS

- Linux: recommandé
- macOS: supporté pour dev/tests locaux
- Windows: supporté via Git Bash ou WSL, pas via `cmd.exe` brut

Le projet est compatible avec `uv` sur les trois OS, mais les dépendances robotiques restent le point sensible sur Windows natif.

## Structure

```text
.
├── robocasa_telecom/
├── scripts/
│   ├── setup_uv.sh
│   ├── with_env.sh
│   ├── run_train.sh
│   ├── run_eval.sh
│   └── slurm/
├── configs/
├── docs/
├── tests/
└── pyproject.toml
```

## Setup

```bash
bash scripts/setup_uv.sh
```

Ce script:
- clone `external/robosuite` et `external/robocasa`;
- fixe les commits utilisés par le projet;
- crée/actualise `.venv` via `uv`;
- installe ce dépôt et les dépendances Python;
- lance les downloads/validations RoboCasa si demandé.

Pour exécuter une commande sans activer manuellement l'environnement:

```bash
scripts/with_env.sh python -m robocasa_telecom.sanity --config configs/env/open_single_door.yaml --steps 20
```

## Vérification

```bash
uv run python -m pip check
pytest -q
uv run python -m robocasa_telecom.sanity --config configs/env/open_single_door.yaml --steps 20
```

## Commandes principales

Train:

```bash
uv run python -m robocasa_telecom.train --config configs/train/open_single_door_ppo.yaml --seed 0
```

Eval:

```bash
uv run python -m robocasa_telecom.evaluate \
  --config configs/train/open_single_door_ppo.yaml \
  --checkpoint checkpoints/<run_id>/final_model.zip \
  --num-episodes 20 \
  --deterministic
```

Visualisation:

```bash
scripts/with_env.sh python scripts/visualize_env.py --config configs/env/open_single_door.yaml --steps 200
```

## SLURM

```bash
sbatch scripts/slurm/train_array.sbatch
sbatch --export=ALL,CHECKPOINT_PATH=checkpoints/<run_id>/final_model.zip scripts/slurm/eval.sbatch
```

## Notes

- `pyproject.toml` est la source de vérité pour le packaging et les dépendances.
- `uv` gère le venv et les dépendances Python, mais pas les paquets système comme `ffmpeg` ou `cmake`.
- les assets RoboCasa restent volumineux et sont téléchargés à part.
