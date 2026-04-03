# RoboCasa Project B (Telecom)

Squelette exécutable pour entraîner et évaluer un agent RL sur RoboCasa, construit autour de `ARISE-Initiative/robosuite` avec une organisation de dépôt proche de Robosuite (package racine + scripts opérationnels + configs + docs + tests).

## Objectif du projet

- Fournir une base propre pour la tâche RoboCasa `OpenCabinet` (alias pédagogiques gérés: `OpenSingleDoor`, `OpenDoor`).
- Garantir une exécution reproductible en local et sur cluster SLURM GPU.
- Produire des artefacts exploitables pour rendu académique: checkpoints, courbes, métriques d'évaluation, logs.

## Pile technique

- Python: `3.11` via Conda.
- Simulation: `robosuite` + `robocasa` installés depuis `external/` en mode editable.
- RL: `stable-baselines3` avec PPO.
- Scheduling cluster: SLURM (`train_array.sbatch`, `eval.sbatch`).

## Structure du dépôt

```text
.
├── robocasa_telecom/           # package Python principal (style robosuite)
│   ├── envs/                   # construction/adaptation des environnements
│   ├── rl/                     # entraînement + évaluation
│   ├── tools/                  # outils de validation (sanity)
│   ├── utils/                  # IO config + utilitaires success
│   ├── train.py                # wrapper CLI: python -m robocasa_telecom.train
│   ├── evaluate.py             # wrapper CLI: python -m robocasa_telecom.evaluate
│   └── sanity.py               # wrapper CLI: python -m robocasa_telecom.sanity
├── scripts/                    # scripts shell / cluster
│   ├── setup_conda.sh
│   ├── run_train.sh
│   ├── run_eval.sh
│   ├── visualize_env.py
│   └── slurm/
├── configs/                    # paramètres YAML env + entraînement
├── docs/                       # documentation détaillée
├── tests/                      # vérifications minimales
├── environment.yml             # env conda de base
└── requirements-project.txt    # dépendances pip projet
```

## Installation

```bash
bash scripts/setup_conda.sh
conda activate robocasa_telecom
```

Si `conda activate` ne marche pas dans votre shell, utilisez directement:

```bash
scripts/with_env.sh python -m robocasa_telecom.sanity --config configs/env/open_single_door.yaml --steps 20
```

Si vous voyez `ModuleNotFoundError: No module named 'robocasa_telecom'`, réinstallez le package projet:

```bash
python -m pip install -e .
```

Installation figée (recommandée projet):

```bash
ENV_NAME=robocasa_telecom \
ROBOCASA_COMMIT=9a3a78680443734786c9784ab661413edb87067b \
ROBOSUITE_COMMIT=aaa8b9b214ce8e77e82926d677b4d61d55e577ab \
DOWNLOAD_ASSETS=1 \
VERIFY_ASSETS=1 \
DOWNLOAD_DATASETS=0 \
bash scripts/setup_conda.sh
```

## Commandes principales

Sanity check installation:

```bash
python -m robocasa_telecom.sanity --config configs/env/open_single_door.yaml --steps 20
```

Entraînement local:

```bash
python -m robocasa_telecom.train --config configs/train/open_single_door_ppo.yaml --seed 0
```

Évaluation checkpoint:

```bash
python -m robocasa_telecom.evaluate \
  --config configs/train/open_single_door_ppo.yaml \
  --checkpoint checkpoints/<run_id>/final_model.zip \
  --num-episodes 20 \
  --deterministic
```

Visualisation environnement:

```bash
scripts/with_env.sh python scripts/visualize_env.py --config configs/env/open_single_door.yaml --steps 200
```

## Exécution cluster SLURM

Lancer un tableau d'entraînement (plusieurs seeds):

```bash
sbatch scripts/slurm/train_array.sbatch
```

Lancer une évaluation:

```bash
sbatch --export=ALL,CHECKPOINT_PATH=checkpoints/<run_id>/final_model.zip scripts/slurm/eval.sbatch
```

## Sorties générées

Par run d'entraînement:
- `checkpoints/<run_id>/`: checkpoints intermédiaires + `final_model.zip`.
- `outputs/<run_id>/monitor.csv`: trace brute SB3 Monitor.
- `outputs/<run_id>/training_curve.csv`: courbe simplifiée prête à tracer.
- `outputs/<run_id>/train_summary.json`: résumé métriques run.

Par run d'évaluation:
- `outputs/eval/eval_YYYYmmdd_HHMMSS.json`: moyennes return/succès.

## Documentation détaillée

- [Architecture détaillée](docs/ARCHITECTURE.md)
- [Méthodes RL et choix techniques](docs/METHODS.md)
- [Runbook local + cluster](docs/RUNBOOK.md)
- [Organisation équipe de 4](docs/COLLABORATION.md)
- [Packages installés/utilisés](docs/PACKAGES.md)
- [CI/CD Linux](docs/CI.md)
- [Référence fichier par fichier](docs/FILE_REFERENCE.md)

## Compatibilité / points d'attention

- Le projet est basé sur Robosuite et réutilise son API `robosuite.make(...)`.
- Certaines installations RoboCasa nécessitent le téléchargement des assets kitchen, sinon `reset()` échoue.
- Le script `scripts/setup_conda.sh` échoue explicitement si `DOWNLOAD_ASSETS=1` mais que les assets critiques ne sont pas présents.
- Le code convertit explicitement les observations pour rester compatible Gymnasium + SB3 selon la version de wrappers disponible.
