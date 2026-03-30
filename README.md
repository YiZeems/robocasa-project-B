# RoboCasa Project B (Telecom) - Architecture alignée Robosuite

Ce dépôt est maintenant structuré pour coller au style de `ARISE-Initiative/robosuite`:
- package Python principal au niveau racine: `robocasa_telecom/`
- `scripts/` pour les entrées opérationnelles
- `configs/` pour les hyperparamètres et paramètres env
- `docs/` pour la documentation technique complète
- `tests/` pour vérifications minimales

## Structure

```text
.
├── robocasa_telecom/
│   ├── envs/
│   ├── rl/
│   ├── tools/
│   ├── utils/
│   ├── train.py
│   ├── evaluate.py
│   └── sanity.py
├── scripts/
│   ├── setup_conda.sh
│   ├── run_train.sh
│   ├── run_eval.sh
│   ├── visualize_env.py
│   └── slurm/
├── configs/
│   ├── env/open_single_door.yaml
│   └── train/open_single_door_ppo.yaml
├── docs/
├── tests/
├── environment.yml
└── requirements-project.txt
```

## Installation

```bash
bash scripts/setup_conda.sh
conda activate robocasa_telecom
```

Variables recommandées:

```bash
ENV_NAME=robocasa_telecom \
ROBOCASA_COMMIT=9a3a78680443734786c9784ab661413edb87067b \
ROBOSUITE_COMMIT=aaa8b9b214ce8e77e82926d677b4d61d55e577ab \
DOWNLOAD_ASSETS=1 \
DOWNLOAD_DATASETS=0 \
bash scripts/setup_conda.sh
```

## Exécution locale

Sanity:

```bash
python -m robocasa_telecom.sanity --config configs/env/open_single_door.yaml --steps 20
```

Train:

```bash
python -m robocasa_telecom.train --config configs/train/open_single_door_ppo.yaml --seed 0
```

Eval:

```bash
python -m robocasa_telecom.evaluate \
  --config configs/train/open_single_door_ppo.yaml \
  --checkpoint checkpoints/<run_id>/final_model.zip \
  --num-episodes 20 --deterministic
```

Visualisation:

```bash
PYTHONPATH=. python scripts/visualize_env.py --config configs/env/open_single_door.yaml --steps 200
```

## Exécution SLURM

```bash
sbatch scripts/slurm/train_array.sbatch
sbatch --export=ALL,CHECKPOINT_PATH=checkpoints/<run_id>/final_model.zip scripts/slurm/eval.sbatch
```

## Documentation détaillée

- [Architecture](docs/ARCHITECTURE.md)
- [Méthodes](docs/METHODS.md)
- [Runbook local + SLURM](docs/RUNBOOK.md)
- [Collaboration Git (groupe de 4)](docs/COLLABORATION.md)

## Compatibilité et task

- Task par défaut: `OpenCabinet`.
- Alias gérés: `OpenSingleDoor`, `OpenDoor`.
- Si les assets RoboCasa manquent, une erreur explicite indique la commande de téléchargement.
