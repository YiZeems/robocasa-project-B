# RoboCasa Telecom - Squelette Projet (Conda + SLURM)

Squelette exécutable pour le projet RoboCasa avec:
- Python 3.11
- conda + pip
- tâche atomique `OpenSingleDoor`
- pipeline `train / eval / sanity`
- jobs batch SLURM (GPU) pour cluster Telecom
- intégration des bonnes pratiques du guide `RoboCasa_Code_Aide.zip` (`GymWrapper(keys=None)`, contrôleur composite, visualisation)

## 1) Prérequis

- Linux avec NVIDIA GPU + drivers OK
- `conda` installé
- accès internet pour cloner `robocasa` / `robosuite`
- cluster SLURM côté Telecom

Optionnel sur cluster:
- charger les modules locaux (exemple): `module load cuda/12.1`

## 2) Installation

```bash
bash scripts/setup_conda.sh
conda activate robocasa_telecom
```

Variables utiles pour setup:

```bash
ENV_NAME=robocasa_telecom \
ROBOCASA_COMMIT=9a3a78680443734786c9784ab661413edb87067b \
ROBOSUITE_COMMIT=aaa8b9b214ce8e77e82926d677b4d61d55e577ab \
DOWNLOAD_ASSETS=0 \
DOWNLOAD_DATASETS=0 \
bash scripts/setup_conda.sh
```

Pour forcer le téléchargement assets/datasets:

```bash
DOWNLOAD_ASSETS=1 DOWNLOAD_DATASETS=1 DATASET_TASK=OpenSingleDoor bash scripts/setup_conda.sh
```

## 3) Sanity Check

```bash
conda activate robocasa_telecom
PYTHONPATH=src python -m robocasa_telecom.sanity --config configs/env/open_single_door.yaml --steps 20
```

## 3bis) Visualisation rapide (inspirée du ZIP guide)

```bash
conda activate robocasa_telecom
PYTHONPATH=src python scripts/visualize_env.py --config configs/env/open_single_door.yaml --steps 200
```

## 4) Entraînement local (test court)

```bash
conda activate robocasa_telecom
PYTHONPATH=src python -m robocasa_telecom.train \
  --config configs/train/open_single_door_ppo.yaml \
  --seed 0 \
  --total-timesteps 5000
```

Sorties:
- `outputs/<run_id>/monitor.csv`
- `outputs/<run_id>/training_curve.csv`
- `outputs/<run_id>/train_summary.json`
- `checkpoints/<run_id>/final_model.zip`

## 5) Évaluation locale

```bash
conda activate robocasa_telecom
PYTHONPATH=src python -m robocasa_telecom.evaluate \
  --config configs/train/open_single_door_ppo.yaml \
  --checkpoint checkpoints/<run_id>/final_model.zip \
  --num-episodes 20 \
  --deterministic
```

## 6) Lancement SLURM (GPU)

### Train en array (1 seed par job)

```bash
sbatch scripts/slurm/train_array.sbatch
```

Personnalisation seeds/config:

```bash
sbatch --export=ALL,SEEDS_CSV=0,1,2,CONFIG_PATH=configs/train/open_single_door_ppo.yaml scripts/slurm/train_array.sbatch
```

### Eval en batch

```bash
sbatch --export=ALL,CHECKPOINT_PATH=checkpoints/<run_id>/final_model.zip scripts/slurm/eval.sbatch
```

## 7) Arborescence

```text
.
├── environment.yml
├── requirements-project.txt
├── configs/
│   ├── env/open_single_door.yaml
│   └── train/open_single_door_ppo.yaml
├── scripts/
│   ├── setup_conda.sh
│   ├── run_train.sh
│   ├── run_eval.sh
│   ├── visualize_env.py
│   └── slurm/
│       ├── train_array.sbatch
│       └── eval.sbatch
└── src/robocasa_telecom/
    ├── env_factory.py
    ├── train.py
    ├── evaluate.py
    └── sanity.py
```

## 8) Notes compatibilité

- Ce squelette exclut `uv` et `pyproject.toml` volontairement.
- Commits figés par défaut:
  - `robocasa`: `9a3a78680443734786c9784ab661413edb87067b`
  - `robosuite`: `aaa8b9b214ce8e77e82926d677b4d61d55e577ab`
- L’env utilise `controller: null` pour laisser RoboSuite charger le contrôleur composite recommandé pour `PandaOmron`.
- Si ton cluster impose une installation PyTorch CUDA spécifique, adapte `requirements-project.txt` ou installe `torch` via l’index recommandé par l’admin cluster.
