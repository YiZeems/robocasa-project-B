# Runbook (local + SLURM)

## 1) Pré-requis

- Linux recommandé pour cluster, macOS/Linux possible en local.
- `conda` accessible en ligne de commande.
- GPU NVIDIA + drivers sur cluster pour les jobs SLURM.
- Accès réseau au moment du setup (clone des dépôts externes et pip install).

## 2) Setup environnement

Commande standard:

```bash
bash scripts/setup_conda.sh
```

Commande explicite (recommandée pour reproductibilité):

```bash
ENV_NAME=robocasa_telecom \
PYTHON_VERSION=3.11 \
ROBOCASA_COMMIT=9a3a78680443734786c9784ab661413edb87067b \
ROBOSUITE_COMMIT=aaa8b9b214ce8e77e82926d677b4d61d55e577ab \
DOWNLOAD_ASSETS=1 \
DOWNLOAD_DATASETS=0 \
RUN_SETUP_MACROS=1 \
bash scripts/setup_conda.sh
```

Activation:

```bash
conda activate robocasa_telecom
```

## 3) Validation rapide installation

```bash
python -m robocasa_telecom.sanity --config configs/env/open_single_door.yaml --steps 20
```

Résultat attendu:
- affichage de la forme de l'observation,
- logs intermédiaires toutes les 5 steps,
- message final `Sanity check completed successfully.`

## 4) Entraînement local

### 4.1 Lancement direct Python

```bash
python -m robocasa_telecom.train \
  --config configs/train/open_single_door_ppo.yaml \
  --seed 0
```

### 4.2 Wrapper shell

```bash
bash scripts/run_train.sh configs/train/open_single_door_ppo.yaml 0
```

Artefacts attendus:
- `checkpoints/<run_id>/final_model.zip`
- `outputs/<run_id>/training_curve.csv`
- `outputs/<run_id>/train_summary.json`

## 5) Évaluation locale

### 5.1 Lancement direct Python

```bash
python -m robocasa_telecom.evaluate \
  --config configs/train/open_single_door_ppo.yaml \
  --checkpoint checkpoints/<run_id>/final_model.zip \
  --num-episodes 20 \
  --deterministic
```

### 5.2 Wrapper shell

```bash
bash scripts/run_eval.sh checkpoints/<run_id>/final_model.zip configs/train/open_single_door_ppo.yaml 20
```

Artefact attendu:
- `outputs/eval/eval_YYYYmmdd_HHMMSS.json`

## 6) Visualisation environnement

```bash
PYTHONPATH=. python scripts/visualize_env.py --config configs/env/open_single_door.yaml --steps 200
```

Usage:
- vérifier rapidement que l'environnement tourne,
- inspecter les rewards et la stabilité des resets.

## 7) Exécution cluster SLURM

## 7.1 Entraînement en array (seeds parallèles)

```bash
sbatch scripts/slurm/train_array.sbatch
```

Variables utiles au submit:

```bash
sbatch --export=ALL,ENV_NAME=robocasa_telecom,CONFIG_PATH=configs/train/open_single_door_ppo.yaml,SEEDS_CSV=0,1,2 scripts/slurm/train_array.sbatch
```

Logs:
- `logs/slurm/robocasa_train-<JOBID>_<TASKID>.out`
- `logs/slurm/robocasa_train-<JOBID>_<TASKID>.err`

## 7.2 Évaluation cluster

```bash
sbatch --export=ALL,CHECKPOINT_PATH=checkpoints/<run_id>/final_model.zip scripts/slurm/eval.sbatch
```

Optionnel:

```bash
sbatch --export=ALL,CHECKPOINT_PATH=checkpoints/<run_id>/final_model.zip,NUM_EPISODES=50 scripts/slurm/eval.sbatch
```

## 8) Dépannage

## Erreur assets RoboCasa manquants

Symptôme:
- `FileNotFoundError` ou erreur XML à l'initialisation d'env.

Correction:

```bash
DOWNLOAD_ASSETS=1 bash scripts/setup_conda.sh
```

ou dans l'environnement actif:

```bash
python -m robocasa.scripts.download_kitchen_assets --type all
```

## Erreur task non trouvée

Le code mappe automatiquement:
- `OpenSingleDoor` -> `OpenCabinet`
- `OpenDoor` -> `OpenCabinet`

## Erreur checkpoint introuvable en eval

Vérifier:
- que `final_model.zip` existe,
- que le chemin passé à `--checkpoint` ou `CHECKPOINT_PATH` est correct,
- que le job SLURM est lancé depuis la racine du dépôt.
