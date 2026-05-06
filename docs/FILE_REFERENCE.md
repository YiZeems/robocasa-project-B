# Référence fichier par fichier

## Fichiers racine

### `pyproject.toml`
Packaging PEP 621, dépendances pinées, entrées console. Dépendances notables : `stable-baselines3==2.3.2`, `imageio-ffmpeg>=0.5` (MP4 sans ffmpeg système), `mlflow==2.20.3`.

### `README.md`
Point d'entrée utilisateur : installation, tableau de support plateforme, démarrage rapide, commandes principales.

### `uv.lock`
Versions exactes de toutes les dépendances (185 packages). Régénérer avec `uv lock` si l'OS cible change (wheels différents Linux/macOS/Windows).

### `Makefile`
Raccourcis : `make train-sac`, `make eval-test`, `make eval-video`, `make plot`, `make check`, `make sanity`.

## Configuration

### `configs/env/open_single_door.yaml`
Paramètres de simulation RoboCasa : tâche, robot, contrôleur, caméras, `max_steps`, `control_freq`.

### `configs/train/open_single_door_sac.yaml`
SAC principal — 3M steps, 12 workers, `device: auto`. Déclare `eval.validation_seed` et `eval.test_seed`.

### `configs/train/open_single_door_sac_debug.yaml`
SAC debug — 300k steps, utilisable pour sanity et smoke tests.

### `configs/train/open_single_door_sac_tuned.yaml`
SAC variante — 2M steps, lr=1e-4, batch=512.

### `configs/train/open_single_door_ppo_baseline.yaml`
PPO baseline — 5M steps, 12 workers, `device: auto`.

### `configs/train/open_single_door_ppo.yaml`
PPO smoke — 200k steps, 4 workers, eval désactivée. Utilisable pour des tests rapides.

## Package principal

### `robocasa_telecom/utils/device.py`
Résolution cross-plateforme du device PyTorch. `resolve_device("auto")` choisit `cuda > mps > cpu`. Contourne le bug SB3 v2.3.x qui ignore MPS sur macOS Apple Silicon. Utilisé par tous les modules RL.

### `robocasa_telecom/envs/factory.py`
Création d'environnement RoboCasa compatible Gymnasium. Auto-détecte `MUJOCO_GL` à l'import (`egl`/`cgl`/`wgl` selon l'OS). Fournit `GymnasiumAdapter` et `RawRoboCasaAdapter` (fallback).

### `robocasa_telecom/rl/train.py`
Boucle d'entraînement PPO/SAC. CLI flags notables : `--n-envs` (workers pairs), `--vec-env subproc|dummy`, `--total-timesteps`, `--algorithm`, `--resume-from`, `--auto-resume`. Inclut `ValidationCallback`, `PeriodicCheckpointCallback`, sauvegarde `best_model.zip`, export `validation_curve.csv` et `train_summary.json`.

### `robocasa_telecom/rl/evaluate.py`
Évaluation d'un checkpoint. Splits `--split validation|test|custom`. Log MLflow avec URI absolue. Export JSON des métriques.

### `robocasa_telecom/rl/eval_video.py`
Vidéo best-episode en deux passes (scoring puis rendu). Single-process obligatoire — les workers SubprocVecEnv ne peuvent pas rendre.

### `robocasa_telecom/rl/render_best_run.py`
Rendu vidéo post-training du meilleur checkpoint. Appelé automatiquement si `ROBOCASA_RENDER_BEST_RUN_VIDEO=1`.

### `robocasa_telecom/utils/io.py`
IO YAML (`load_yaml`) + création de dossiers (`ensure_dir`).

### `robocasa_telecom/utils/success.py`
Détection homogène du succès de tâche (compatible toutes versions RoboCasa).

### `robocasa_telecom/utils/metrics.py`
Calcul de métriques RL : `door_angle`, `action_magnitude`, métriques anti-hacking (oscillation, stagnation, `approach_frac`).

### `robocasa_telecom/utils/checkpoints.py`
Résolution et sauvegarde des artefacts checkpoint. Supporte reprise automatique (`find_latest_resume_candidate`).

### `robocasa_telecom/utils/video.py`
Helpers MP4 : `grid_2x2` (mosaïque 4 caméras), `ensure_uint8_frame`, `save_mp4` (via imageio-ffmpeg).

### `robocasa_telecom/sanity.py`
Smoke test minimal : reset + N steps aléatoires. Valide l'installation sans training.

## Tests

### `tests/test_platform_smoke.py`
11 tests cross-plateforme sans GPU ni MuJoCo. Vérifie : `resolve_device`, `MUJOCO_GL` auto-set, `pathlib`, URI MLflow absolue, imports imageio-ffmpeg/SB3/torch. Lancer avant tout push : `pytest tests/test_platform_smoke.py -v`.

### `tests/test_config_loading.py`
Test de cohérence des configs YAML.

### `tests/test_env_factory.py`
Test de la factory d'environnement.

### `tests/test_metrics.py`
Test des utilitaires de métriques.

### `tests/test_checkpoints.py`
Test de la gestion des checkpoints.

### `tests/test_train_video_env.py`
Test de l'environnement de rendu vidéo.

## Scripts

### `scripts/setup_uv.sh`
Bootstrap complet : clone des externals (commits figés), `uv sync`, liaison/téléchargement des assets RoboCasa, validation d'imports.

### `scripts/run_train.sh` / `scripts/run_eval.sh`
Wrappers locaux. La voie standard reste `uv run`.

### `scripts/slurm/train_array.sbatch`
Lancement batch train sur cluster SLURM (array par seed).

### `scripts/slurm/eval.sbatch`
Lancement batch évaluation sur cluster.

### `scripts/slurm/render_best_run.sbatch`
Lancement rendu vidéo sur cluster.

## Documentation

### `docs/RUNBOOK.md`
Procédures opératoires : setup, validation, train, eval, vidéo, MLflow, SLURM, dépannage rapide.

### `docs/ARCHITECTURE.md`
Architecture en couches, flux d'exécution détaillé, device resolution, VecEnv, reproductibilité.

### `docs/platform_compatibility.md`
Instructions complètes par OS (macOS/Windows/WSL2) : prérequis, install, commandes de test, variables d'environnement.

### `docs/EXPERIMENTS.md`
Plan de runs : SAC debug/principal/tuned + PPO baseline. Diagramme boucle entraînement.

### `docs/reproducibility.md`
Guide complet reproductibilité : versions, seeds, déterminisme par hardware, stratégie artefacts Git.

### `docs/troubleshooting.md`
Résolution des problèmes courants : install, MLflow, vidéo, rendu, seeds, device.

### `docs/METHODS.md`
Justification des choix algorithmiques (SAC vs PPO, reward shaping, anti-hacking).

### `docs/metrics.md`
Description des métriques collectées et de leur interprétation.

### `docs/reward_shaping.md`
Détail du reward shaping et des termes anti-hacking.

### `docs/CI.md`
Synthèse CI/CD.

### `docs/PACKAGES.md`
Inventaire et justification des dépendances.

## Historique

Les exports dans `docs/packages/` sont conservés comme historique de l'environnement.
