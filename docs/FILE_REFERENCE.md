# Référence fichier par fichier

Ce document décrit, pour chaque fichier versionné du projet, son rôle, les méthodes utilisées et ses interfaces principales.

## Fichiers racine

### `.gitignore`
- Rôle: exclure artefacts locaux (`__pycache__`, logs, outputs, checkpoints, dépôts externes).
- Méthode: règles de pattern Git standard.
- Impact: garde l'historique Git propre et reproductible.

### `environment.yml`
- Rôle: base Conda (`python=3.11`, outils build essentiels).
- Méthode: spécification YAML Conda.
- Entrée: utilisée indirectement par l'environnement créé dans `setup_conda.sh`.
- Sortie: environnement runtime compatible.

### `requirements-project.txt`
- Rôle: dépendances pip du code projet (RL, config, logging/plot).
- Méthode: pins ou bornes de versions pour limiter les conflits.
- Entrée: consommé par `pip install -r`.

### `README.md`
- Rôle: point d'entrée utilisateur (installation, exécution, structure, docs).
- Méthode: documentation opérationnelle concise.

## Configuration

### `configs/env/open_single_door.yaml`
- Rôle: paramètres de simulation RoboCasa.
- Méthode: déclaration YAML `env:` -> convertie en `EnvConfig`.
- Paramètres clés: `task`, `robots`, horizon, caméra, renderer.

### `configs/train/open_single_door_ppo.yaml`
- Rôle: hyperparamètres PPO + chemins de sortie.
- Méthode: YAML `project/paths/env/train`.
- Paramètres clés: `total_timesteps`, `n_steps`, `batch_size`, `learning_rate`, `save_freq_steps`.

## Package principal `robocasa_telecom/`

### `robocasa_telecom/__init__.py`
- Rôle: point d'entrée package.
- Méthode: expose les sous-packages publics.

### `robocasa_telecom/train.py`
- Rôle: wrapper CLI stable (`python -m robocasa_telecom.train`).
- Méthode: délégation vers `robocasa_telecom.rl.train.main`.

### `robocasa_telecom/evaluate.py`
- Rôle: wrapper CLI stable pour l'évaluation.
- Méthode: délégation vers `robocasa_telecom.rl.evaluate.main`.

### `robocasa_telecom/sanity.py`
- Rôle: wrapper CLI stable pour le sanity check.
- Méthode: délégation vers `robocasa_telecom.tools.sanity.main`.

### `robocasa_telecom/env_factory.py`
- Rôle: compatibilité import historique.
- Méthode: re-export de `EnvConfig`, `load_env_config`, `make_env_from_config`.

### `robocasa_telecom/config_utils.py`
- Rôle: compatibilité import utilitaires config.
- Méthode: re-export de `load_yaml`, `ensure_dir`.

### `robocasa_telecom/success_utils.py`
- Rôle: compatibilité import utilitaire succès.
- Méthode: re-export de `infer_success`.

## Sous-package `robocasa_telecom/envs`

### `robocasa_telecom/envs/__init__.py`
- Rôle: expose les API de création d'environnement.

### `robocasa_telecom/envs/factory.py`
- Rôle: cœur de création environnement RoboCasa.
- Méthodes/classes:
  - `EnvConfig`: dataclass de paramètres env.
  - `GymnasiumAdapter`: adapte wrapper robosuite à API Gymnasium.
  - `RawRoboCasaAdapter`: fallback robuste, flatten observations dict.
  - `_resolve_controller_config`: stratégie fallback contrôleur.
  - `load_env_config`: parse YAML -> `EnvConfig`.
  - `make_env_from_config`: instancie `robosuite.make(...)`, gère alias tâches et erreurs assets.
- Entrées: config YAML + seed.
- Sorties: env Gymnasium-compatible pour SB3.

## Sous-package `robocasa_telecom/rl`

### `robocasa_telecom/rl/__init__.py`
- Rôle: index des entrées RL.

### `robocasa_telecom/rl/train.py`
- Rôle: pipeline complet d'entraînement PPO.
- Méthodes:
  - `parse_args`: interface CLI train.
  - `_evaluate_policy`: évaluation courte post-train.
  - `_export_training_curve`: conversion `monitor.csv` -> CSV simplifié.
  - `main`: orchestration complète (config, env, modèle, learn, exports).
- Entrées: config train, seed, timesteps.
- Sorties: checkpoints, métriques JSON, courbe CSV.

### `robocasa_telecom/rl/evaluate.py`
- Rôle: évaluation d'un checkpoint PPO.
- Méthodes:
  - `parse_args`: interface CLI eval.
  - `main`: rollouts + agrégation + export JSON.
- Entrées: config train + chemin checkpoint.
- Sorties: métriques d'évaluation dans `outputs/eval`.

## Sous-package `robocasa_telecom/tools`

### `robocasa_telecom/tools/__init__.py`
- Rôle: expose les outils runtime.

### `robocasa_telecom/tools/sanity.py`
- Rôle: test fonctionnel minimal de l'environnement.
- Méthodes:
  - `parse_args`: CLI sanity.
  - `main`: reset + random steps + fermeture propre.
- Entrée: config env.
- Sortie: logs console.

## Sous-package `robocasa_telecom/utils`

### `robocasa_telecom/utils/__init__.py`
- Rôle: expose utilitaires transverses.

### `robocasa_telecom/utils/io.py`
- Rôle: fonctions IO réutilisables.
- Méthodes:
  - `load_yaml`: lecture YAML sécurisée + validation mapping racine.
  - `ensure_dir`: création idempotente de dossier.

### `robocasa_telecom/utils/success.py`
- Rôle: extraction robuste du succès d'épisode.
- Méthode:
  - `infer_success`: inspection `info` puis fallback `_check_success` en remontant les wrappers.

## Scripts d'exécution

### `scripts/setup_conda.sh`
- Rôle: bootstrap complet environnement.
- Méthode:
  - création env conda,
  - clone `robosuite` / `robocasa` dans `external/`,
  - checkout commits fixés,
  - install editable + requirements pip,
  - setup macros et téléchargements (assets/datasets) pilotés par variables.
- Entrées: variables d'environnement (`ENV_NAME`, commits, flags download...).

### `scripts/run_train.sh`
- Rôle: wrapper local homogène pour train.
- Méthode: activation conda + lancement module train avec `PYTHONPATH` repo.

### `scripts/run_eval.sh`
- Rôle: wrapper local homogène pour eval.
- Méthode: validation checkpoint + activation conda + lancement module eval.

### `scripts/visualize_env.py`
- Rôle: smoke test visuel/offscreen multi-caméra.
- Méthode: force `use_camera_obs` + random policy.

## Scripts cluster

### `scripts/slurm/train_array.sbatch`
- Rôle: paralléliser plusieurs seeds sur GPU (array jobs).
- Méthode:
  - mapping `SLURM_ARRAY_TASK_ID` -> seed via `SEEDS_CSV`,
  - activation conda,
  - lancement du module train.
- Sorties: logs par tâche + artefacts séparés par run/seed.

### `scripts/slurm/eval.sbatch`
- Rôle: exécuter une évaluation batch d'un checkpoint.
- Méthode: lit `CHECKPOINT_PATH`, lance `robocasa_telecom.evaluate`.
- Sorties: logs + JSON de métriques dans `outputs/eval`.

## Tests

### `tests/test_config_loading.py`
- Rôle: test minimal de cohérence des configs.
- Méthode:
  - charge config env,
  - vérifie robot et nom de tâche attendu,
  - imprime un message de succès.

## Documentation

### `docs/ARCHITECTURE.md`
- Rôle: vue couches + flux d'exécution + extensibilité.

### `docs/METHODS.md`
- Rôle: description des choix RL, du pipeline et des limites.

### `docs/RUNBOOK.md`
- Rôle: procédures opératoires local/cluster + dépannage.

### `docs/COLLABORATION.md`
- Rôle: modèle de collaboration Git en équipe de 4.

### `docs/PACKAGES.md`
- Rôle: inventaire des packages installés et packages utilisés avec explication de leur rôle.

### `docs/FILE_REFERENCE.md`
- Rôle: index technique détaillé fichier par fichier (ce document).

### `docs/packages/conda_list_export_2026-03-30.txt`
- Rôle: export exhaustif de l'environnement Conda (versions exactes, packages installés).

### `docs/packages/pip_freeze_2026-03-30.txt`
- Rôle: export exhaustif `pip freeze` de l'environnement projet.
