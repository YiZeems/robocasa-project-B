# Référence fichier par fichier

## Fichiers racine

### `pyproject.toml`
- Rôle: définit le packaging, les dépendances, et les entrées console.
- Méthode: metadata PEP 621 + backend setuptools.

### `README.md`
- Rôle: point d'entrée utilisateur pour installation et usage.

## Configuration

### `configs/env/open_single_door.yaml`
- Rôle: paramètres de simulation RoboCasa.

### `configs/train/open_single_door_ppo.yaml`
- Rôle: hyperparamètres PPO et sorties.

## Package principal

### `robocasa_telecom/train.py`
- Rôle: wrapper CLI train.

### `robocasa_telecom/evaluate.py`
- Rôle: wrapper CLI eval.

### `robocasa_telecom/sanity.py`
- Rôle: wrapper CLI sanity.

### `robocasa_telecom/envs/factory.py`
- Rôle: création d'environnement RoboCasa compatible Gymnasium.

## Scripts

### `scripts/setup_uv.sh`
- Rôle: bootstrap complet de l'environnement via `uv`.
- Méthode: clone des externals, `uv sync`, installation editable des repos externes, téléchargement éventuel des assets.

### `scripts/with_env.sh`
- Rôle: exécute une commande dans l'environnement `uv` du projet.

### `scripts/run_train.sh`
- Rôle: wrapper local pour l'entraînement.

### `scripts/run_eval.sh`
- Rôle: wrapper local pour l'évaluation.

### `scripts/slurm/train_array.sbatch`
- Rôle: lancement batch train sur cluster.

### `scripts/slurm/eval.sbatch`
- Rôle: lancement batch eval sur cluster.

## Tests

### `tests/test_config_loading.py`
- Rôle: test minimal de cohérence de config.

## Documentation

### `docs/RUNBOOK.md`
- Rôle: procédures opératoires.

### `docs/CI.md`
- Rôle: synthèse CI/CD.

### `docs/PACKAGES.md`
- Rôle: inventaire des dépendances.

## Historique

- Les exports Conda/Pip dans `docs/packages/` sont conservés comme historique.
