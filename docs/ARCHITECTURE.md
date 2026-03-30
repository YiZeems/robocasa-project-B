# Architecture

## Référence de design

Architecture alignée avec le style `robosuite`:
- package principal au niveau racine (`robocasa_telecom/`)
- logique métier structurée par sous-domaines
- scripts d'entrée séparés du cœur métier

## Modules

## `robocasa_telecom/envs`

Responsabilités:
- construction des environnements RoboCasa / RoboSuite
- adaptation API vers interface Gymnasium compatible SB3
- gestion des alias de tâches (`OpenSingleDoor` -> `OpenCabinet`)
- gestion explicite des erreurs d'assets manquants

Fichier principal:
- `factory.py`

## `robocasa_telecom/rl`

Responsabilités:
- entraînement PPO (`train.py`)
- évaluation checkpoint (`evaluate.py`)
- export artefacts (`train_summary.json`, `training_curve.csv`, `eval_*.json`)

## `robocasa_telecom/tools`

Responsabilités:
- outils de validation runtime (`sanity.py`)

## `robocasa_telecom/utils`

Responsabilités:
- IO YAML et création de dossiers (`io.py`)
- extraction robuste du signal de succès (`success.py`)

## Wrappers compatibilité

Pour ne pas casser les commandes historiques:
- `robocasa_telecom.train`
- `robocasa_telecom.evaluate`
- `robocasa_telecom.sanity`
- `robocasa_telecom.env_factory`
- `robocasa_telecom.config_utils`
- `robocasa_telecom.success_utils`

## Répertoires opérationnels

- `configs/`: paramètres env + entraînement
- `scripts/`: setup, run local, scripts sbatch
- `docs/`: docs techniques et organisationnelles
- `tests/`: validations simples de configuration/import
