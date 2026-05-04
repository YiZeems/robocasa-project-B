# Architecture détaillée

## 1) Principes d'architecture

Le dépôt suit une logique proche de `robosuite`:
- un package Python principal à la racine (`robocasa_telecom/`),
- des scripts d'orchestration séparés (`scripts/`),
- des configurations déclaratives (`configs/`),
- des documents opératoires (`docs/`).

Cette séparation permet:
- de garder la logique métier testable hors cluster,
- d'avoir des entrées CLI simples pour l'utilisateur,
- de limiter l'impact quand on remplace une méthode RL ou une tâche RoboCasa.

## 2) Vue des couches

### Couche Configuration
- `configs/env/open_single_door.yaml`: paramètres de la simulation RoboCasa.
- `configs/train/open_single_door_ppo.yaml`: hyperparamètres RL + chemins d'artefacts.

### Couche Environnement
- `robocasa_telecom/envs/factory.py`: traduit la config YAML en env RoboCasa exécutable.
- Fournit deux adaptateurs Gymnasium:
  - `GymnasiumAdapter` (seulement si le `GymWrapper` RoboSuite garde une shape stable),
  - `RawRoboCasaAdapter` (fallback robuste, flatten dict observations + normalisation de taille).

### Couche RL
- `robocasa_telecom/rl/train.py`: boucle d'entraînement PPO + checkpoints + export courbes.
- `robocasa_telecom/rl/evaluate.py`: chargement checkpoint + rollouts d'évaluation.

### Couche Outils
- `robocasa_telecom/tools/sanity.py`: test minimal import/reset/step pour valider l'install.

### Couche Utilitaires
- `robocasa_telecom/utils/io.py`: IO YAML + création dossiers.
- `robocasa_telecom/utils/success.py`: logique homogène de détection du succès.

### Couche Exécution
- `scripts/setup_uv.sh`: provisioning environnement.
- `scripts/run_train.sh` / `scripts/run_eval.sh`: wrappers locaux.
- `scripts/slurm/*.sbatch`: exécution GPU cluster.

## 3) Flux d'exécution détaillé

### 3.1 Train
1. `uv run python -m robocasa_telecom.train ...`
2. wrapper `robocasa_telecom/train.py` -> `robocasa_telecom/rl/train.py`.
3. Chargement YAML train + YAML env.
4. Création env via `make_env_from_config`.
5. Wrapping Monitor SB3 (écriture `monitor.csv`).
6. Construction modèle PPO ou SAC.
7. `model.learn(...)` avec `PeriodicCheckpointCallback` + `ValidationCallback`.
8. Sauvegarde `final_model.zip` + métadonnées de reprise.
9. Évaluation post-train + log MLflow.
10. Export `training_curve.csv`, `validation_curve.csv` et `train_summary.json`.

### 3.2 Évaluation
1. `uv run python -m robocasa_telecom.evaluate ...`
2. Chargement config + checkpoint.
3. Rollouts déterministes/non déterministes.
4. Calcul `return_mean`, `return_std`, `success_rate`.
5. Export JSON dans `outputs/eval/`.

### 3.3 Sanity
1. `uv run python -m robocasa_telecom.sanity ...`
2. Reset env.
3. `N` pas aléatoires.
4. Succès si aucune exception + fermeture propre.

## 4) Gestion de compatibilité robosuite / robocasa

- Le code importe `robocasa` avant `robosuite.make(...)` pour enregistrer les tâches RoboCasa.
- Les alias `OpenSingleDoor` et `OpenDoor` sont convertis en `OpenCabinet`.
- Le contrôleur est résolu avec fallback:
  - `load_composite_controller_config`,
  - sinon `load_part_controller_config`,
  - sinon composite par défaut.
- Le `GymWrapper` est sondé sur plusieurs resets; s'il dérive, le projet retombe automatiquement sur `RawRoboCasaAdapter`.
- Si les assets manquent, l'erreur est transformée en message actionnable.

## 5) Artefacts et reproductibilité

- Le nom de run inclut `task`, `seed`, timestamp.
- Les chemins d'artefacts sont centralisés dans `configs/train/*.yaml`.
- Les seeds sont explicitement injectées en local et en SLURM array.
- Les commits `robosuite` / `robocasa` sont fixables dans `setup_uv.sh`.

## 6) Extensibilité

Pour ajouter une nouvelle tâche:
1. dupliquer `configs/env/open_single_door.yaml`,
2. changer `env.task`,
3. dupliquer la config train correspondante,
4. lancer train/eval avec la nouvelle config.

Pour ajouter un nouvel algo:
1. créer un module `robocasa_telecom/rl/<algo>.py`,
2. conserver les interfaces CLI (`--config`, `--seed`, etc.),
3. réutiliser `envs.factory` et `utils.success` pour homogénéité.
