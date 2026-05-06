# Runbook

## Pré-requis

| Plateforme | Pré-requis supplémentaires |
|---|---|
| macOS Apple Silicon | `brew install cmake` |
| Linux / WSL2 | Drivers NVIDIA + CUDA, `sudo apt install cmake libgl1-mesa-glx` |
| Windows 11 | Visual C++ Build Tools 2022, CUDA Toolkit 12.x |

Voir [`docs/platform_compatibility.md`](platform_compatibility.md) pour les instructions complètes par OS.

Commun à toutes les plateformes : `uv` dans le `PATH`, `git`, accès réseau au moment du setup.

## Setup

```bash
bash scripts/setup_uv.sh
```

Exemple complet :

```bash
PYTHON_VERSION=3.11 \
ROBOCASA_COMMIT=9a3a78680443734786c9784ab661413edb87067b \
ROBOSUITE_COMMIT=aaa8b9b214ce8e77e82926d677b4d61d55e577ab \
DOWNLOAD_ASSETS=1 \
VERIFY_ASSETS=1 \
DOWNLOAD_DATASETS=0 \
RUN_SETUP_MACROS=1 \
bash scripts/setup_uv.sh
```

Ce setup :

- clone `external/robosuite` et `external/robocasa`,
- installe le projet avec `uv sync`,
- installe `robocasa` et `robosuite` depuis les sources locales déclarées dans `pyproject.toml`,
- relie automatiquement `robocasa/models/assets` vers les assets du checkout externe si besoin,
- vérifie les imports et la présence d'assets critiques.

## Validation

```bash
# Smoke tests cross-plateforme (aucun GPU ni MuJoCo requis, < 5 s)
pytest tests/test_platform_smoke.py -v

# Vérification de l'environnement
make check
uv run python -m robocasa_telecom.sanity --config configs/env/open_single_door.yaml --steps 20
```

Smoke train recommandé :

```bash
# Mode single-process (sûr sur toutes les plateformes, y compris Windows)
uv run python -m robocasa_telecom.train \
  --config configs/train/open_single_door_sac_debug.yaml \
  --seed 0 \
  --total-timesteps 10 \
  --vec-env dummy \
  --no-auto-resume

# Mode parallèle (SubprocVecEnv, fork — Linux/WSL2)
uv run python -m robocasa_telecom.train \
  --config configs/train/open_single_door_sac_debug.yaml \
  --seed 0 \
  --total-timesteps 10 \
  --n-envs 2 \
  --no-auto-resume
```

## Train

```bash
uv run python -m robocasa_telecom.train --config configs/train/open_single_door_sac.yaml --seed 0
```

Ou via le wrapper shell :

```bash
bash scripts/run_train.sh configs/train/open_single_door_ppo.yaml 0
```

Reprise d'un run interrompu :

```bash
uv run python -m robocasa_telecom.train \
  --config configs/train/open_single_door_sac.yaml \
  --seed 0 \
  --resume-from checkpoints/<run_id>/sac_100000_steps.zip
```

Sans `--resume-from`, l'entraînement active `--auto-resume` par défaut et reprend le dernier run interrompu correspondant au même `task/algo/seed`. Utilise `--no-auto-resume` pour forcer un départ depuis zéro.

### Options CLI notables

| Flag | Valeurs | Effet |
|---|---|---|
| `--n-envs N` | entier pair ≥ 2 | override le nombre de workers parallèles |
| `--vec-env` | `subproc` (défaut) / `dummy` | backend VecEnv ; `dummy` = single-process (debug) |
| `--total-timesteps N` | entier | override la durée de l'entraînement |
| `--algorithm` | `SAC` / `PPO` | override l'algo déclaré dans le YAML |
| `--seed N` | entier | override le seed |

### Vidéo post-training

Pour générer automatiquement une vidéo 4 vues du meilleur checkpoint à la fin du `train` :

```bash
ROBOCASA_RENDER_BEST_RUN_VIDEO=1 uv run python -m robocasa_telecom.train \
  --config configs/train/open_single_door_sac.yaml --seed 0
```

La vidéo est écrite sous `outputs/<run_id>/videos/`. Variables d'ajustement :

```bash
ROBOCASA_RENDER_BEST_RUN_VIDEO_MAX_STEPS=600
ROBOCASA_RENDER_BEST_RUN_VIDEO_FPS=20
ROBOCASA_RENDER_BEST_RUN_VIDEO_MIN_SECONDS=12
ROBOCASA_RENDER_BEST_RUN_VIDEO_MAX_EPISODES=5
```

### Configs disponibles

```bash
uv run python -m robocasa_telecom.train --config configs/train/open_single_door_sac_debug.yaml --seed 0    # 300k — sanity
uv run python -m robocasa_telecom.train --config configs/train/open_single_door_sac.yaml --seed 0          # 3M   — run principal
uv run python -m robocasa_telecom.train --config configs/train/open_single_door_sac_tuned.yaml --seed 0    # 2M   — variante
uv run python -m robocasa_telecom.train --config configs/train/open_single_door_ppo.yaml --seed 0          # 200k — smoke PPO
uv run python -m robocasa_telecom.train --config configs/train/open_single_door_ppo_baseline.yaml --seed 0 # 5M   — baseline PPO
```

## Eval

```bash
uv run python -m robocasa_telecom.evaluate \
  --config configs/train/open_single_door_sac.yaml \
  --checkpoint checkpoints/<run_id>/best_model.zip \
  --num-episodes 50 \
  --split test \
  --deterministic
```

Splits disponibles : `--split validation` (seed=10000), `--split test` (seed=20000), `--split custom --seed N`.

## Vidéo best-episode

```bash
uv run python -m robocasa_telecom.rl.eval_video \
  --config configs/train/open_single_door_sac.yaml \
  --checkpoint checkpoints/<run_id>/best_model.zip \
  --n-episodes 20 \
  --seed 0 \
  --output-dir outputs/eval/videos/
```

## Render best run (post-training)

```bash
uv run robocasa-telecom-render-best-run \
  --config configs/train/open_single_door_sac.yaml \
  --checkpoint checkpoints/<run_id>/best_model.zip
```

## MLflow

```bash
# macOS / Linux / WSL2
uv run mlflow ui --backend-store-uri $(realpath mlruns)

# Windows PowerShell
uv run mlflow ui --backend-store-uri (Resolve-Path mlruns).Path
```

Le tracking URI est ancré en chemin absolu (`file://`) à l'initialisation — `mlflow ui` fonctionne depuis n'importe quel répertoire.

## SLURM

```bash
sbatch scripts/slurm/train_array.sbatch
sbatch --export=ALL,CHECKPOINT_PATH=checkpoints/<run_id>/final_model.zip scripts/slurm/eval.sbatch
sbatch --export=ALL,CONFIG_PATH=configs/train/open_single_door_sac.yaml,CHECKPOINT_PATH=checkpoints/<run_id>/best_model.zip,SEED=0 scripts/slurm/render_best_run.sbatch
```

## Dépannage rapide

| Symptôme | Solution |
|---|---|
| `uv run` échoue | `.venv` absent — relancer `bash scripts/setup_uv.sh` |
| Assets RoboCasa manquants | `DOWNLOAD_ASSETS=1 VERIFY_ASSETS=1 bash scripts/setup_uv.sh` |
| Shape d'observation instable | Normal — bascule automatique sur `RawRoboCasaAdapter` |
| `MUJOCO_GL` non défini / render vide | Auto-défini par `factory.py` (egl/cgl/wgl) ; forcer avec `MUJOCO_GL=osmesa` si nécessaire |
| `SubprocVecEnv` / pickling sur Windows | Utiliser `--vec-env dummy` pour passer en single-process |
| Device inattendu (`cpu` au lieu de `mps`) | `resolve_device()` dans `utils/device.py` — vérifier avec `python -c "from robocasa_telecom.utils.device import resolve_device; print(resolve_device('auto'))"` |
| Port 5000 occupé (MLflow) | `uv run mlflow ui --port 5001` |

Voir [`docs/troubleshooting.md`](troubleshooting.md) et [`docs/platform_compatibility.md`](platform_compatibility.md) pour les cas détaillés.
