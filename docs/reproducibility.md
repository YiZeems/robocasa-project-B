# Reproductibilité — Guide complet

---

## 1. Environnement testé

| Composant | Version | Notes |
|---|---|---|
| OS | macOS 15.x (Apple Silicon M-series) | Aussi compatible Ubuntu 22.04, Windows 11 |
| Python | 3.11 | Fixé dans `pyproject.toml` |
| MuJoCo | 3.x (via robosuite) | Installé automatiquement |
| Stable-Baselines3 | 2.3.2 | Fixé dans `pyproject.toml` |
| Gymnasium | 0.29.1 | Fixé dans `pyproject.toml` |
| RoboCasa | 1.0.0 | Commit figé dans `setup_uv.sh` |
| RoboSuite | 1.5.2 | Commit figé dans `setup_uv.sh` |
| MLflow | 2.20.3 | Fixé dans `pyproject.toml` |
| imageio-ffmpeg | ≥ 0.5 | MP4 sans ffmpeg système (Windows/macOS) |
| uv | ≥ 0.4 | Gestionnaire de paquets |

---

## 2. Installation reproductible

### Étape 1 — Cloner le repo

```bash
git clone https://github.com/YiZeems/robocasa-project-B.git
cd robocasa-project-B
# Optionnel : se placer sur un commit spécifique pour la reproductibilité parfaite
# git checkout <commit_sha>
```

### Étape 2 — Installation complète

```bash
PYTHON_VERSION=3.11 \
DOWNLOAD_ASSETS=1 VERIFY_ASSETS=1 \
DOWNLOAD_DATASETS=0 RUN_SETUP_MACROS=1 \
bash scripts/setup_uv.sh
```

Ce script :
- Clone `external/robosuite` au commit `aaa8b9b214ce8e77e82926d677b4d61d55e577ab`
- Clone `external/robocasa` au commit `9a3a78680443734786c9784ab661413edb87067b`
- Crée `.venv` via `uv` avec Python 3.11
- Installe toutes les dépendances depuis `uv.lock` (versions exactes)
- Télécharge et vérifie les assets RoboCasa

### Étape 3 — Vérification

```bash
pytest tests/test_platform_smoke.py -v   # 11 tests cross-plateforme, < 5 s, sans GPU
make check    # compile + vérification des imports
make sanity   # smoke test 20 steps reset/step
```

---

## 3. Seeds et déterminisme

### Seeds utilisés

| Seed | Usage | Valeur par défaut |
|---|---|---|
| `train.seed` | Seed d'entraînement (policy init, replay buffer) | 0 |
| `eval.validation_seed` | Seed pour les évaluations périodiques pendant le training | 10 000 |
| `eval.test_seed` | Seed pour l'évaluation finale (split test) | 20 000 |
| `base_seed` (eval_video) | Seed de base pour les N épisodes scorés | 0 |

**Pourquoi des seeds séparés ?** Le split validation (seed=10000) et le split test (seed=20000) garantissent que l'agent n'a jamais vu les configurations initiales du split test pendant le training ou la sélection de checkpoint.

### Niveau de déterminisme

Sur CPU, les runs sont **entièrement reproductibles** avec le même seed. Sur GPU (CUDA), des variations minimes peuvent apparaître dues aux opérations non-déterministes de cuDNN. Sur MPS (Apple Silicon), le déterminisme est intermédiaire.

Pour un déterminisme maximum :

```bash
PYTHONHASHSEED=0 uv run python -m robocasa_telecom.train \
  --config configs/train/open_single_door_sac.yaml \
  --seed 0 --device cpu
```

---

## 4. Reproduire un run complet

### SAC principal (3M steps, seed=0)

```bash
# 1. Installation
bash scripts/setup_uv.sh

# 2. Entraînement
uv run python -m robocasa_telecom.train \
  --config configs/train/open_single_door_sac.yaml \
  --seed 0

# 3. Évaluation sur le split test
make eval-test \
  CONFIG=configs/train/open_single_door_sac.yaml \
  CHECKPOINT=checkpoints/<run_id>/best_model.zip \
  EPISODES=50

# 4. Génération vidéo
make eval-video \
  CONFIG=configs/train/open_single_door_sac.yaml \
  CHECKPOINT=checkpoints/<run_id>/best_model.zip \
  EPISODES=20 SEED=0

# 5. Courbes
make plot PLOT_RUNS="outputs/OpenCabinet_SAC_seed0_*/"

# 6. MLflow UI
uv run mlflow ui --backend-store-uri ./mlruns
```

### PPO baseline (5M steps, seed=0)

```bash
uv run python -m robocasa_telecom.train \
  --config configs/train/open_single_door_ppo_baseline.yaml \
  --seed 0

make eval-test \
  CONFIG=configs/train/open_single_door_ppo_baseline.yaml \
  CHECKPOINT=checkpoints/<run_id>/best_model.zip \
  EPISODES=50
```

---

## 5. Reproduire les courbes

```bash
# Courbes d'un seul run
uv run python scripts/plot_training.py \
  --run outputs/OpenCabinet_SAC_seed0_<timestamp>/ \
  --out outputs/plots/ \
  --smooth 3

# Comparaison SAC vs PPO
uv run python scripts/plot_training.py \
  --run outputs/OpenCabinet_SAC_seed0_*/ outputs/OpenCabinet_PPO_seed0_*/ \
  --label "SAC" "PPO" \
  --out outputs/plots/ \
  --smooth 3
```

Fichiers générés :

```text
outputs/plots/
  success_rate.png     ← val_success_rate vs timesteps (avec CI 95%)
  return_mean.png      ← val_return_mean vs timesteps
  door_angle.png       ← door_angle_final + door_angle_max
  anti_hacking.png     ← approach_frac, stagnation, sign_changes
  reward_components.png ← composantes de reward au fil du temps
  summary.png          ← grille 2×3 des 6 métriques principales
```

---

## 6. Reproduire la vidéo

```bash
make eval-video \
  CONFIG=configs/train/open_single_door_sac.yaml \
  CHECKPOINT=checkpoints/<run_id>/best_model.zip \
  EPISODES=20 SEED=0 VIDEO_OUT=outputs/eval/videos/
```

La vidéo est reproductible : avec le même checkpoint, le même seed et la même config, `eval_video.py` produit exactement les mêmes trajectoires (politique déterministe + seed fixé).

---

## 7. Fichiers de reproductibilité par run

Pour chaque run, les fichiers suivants permettent de reproduire exactement les résultats :

| Fichier | Contenu | Usage |
|---|---|---|
| `outputs/<run_id>/resolved_train_config.yaml` | Config exacte utilisée (tous overrides résolus) | Reproduire avec la même config |
| `outputs/<run_id>/train_summary.json` | Résumé complet : seed, steps, meilleur checkpoint, métriques | Vérifier les résultats |
| `outputs/<run_id>/validation_curve.csv` | Toutes les métriques validation step par step | Reproduire les courbes |
| `checkpoints/<run_id>/sac_<step>_steps.json` | Métadonnées du checkpoint (num_timesteps, run_id) | Reprendre depuis ce checkpoint |
| `uv.lock` | Versions exactes de toutes les dépendances | Reproduire l'environnement |

---

## 8. Reprendre un run interrompu

```bash
# Reprise automatique (détecte le dernier run incomplet compatible)
uv run python -m robocasa_telecom.train \
  --config configs/train/open_single_door_sac.yaml \
  --seed 0

# Reprise explicite depuis un checkpoint
uv run python -m robocasa_telecom.train \
  --config configs/train/open_single_door_sac.yaml \
  --seed 0 \
  --resume-from checkpoints/<run_id>/sac_1000000_steps.zip

# Forcer un départ depuis zéro (ignore l'auto-resume)
uv run python -m robocasa_telecom.train \
  --config configs/train/open_single_door_sac.yaml \
  --seed 0 --no-auto-resume
```

Pour SAC, le replay buffer est sauvegardé avec chaque checkpoint périodique (`sac_<step>_steps_replay_buffer.pkl`). La reprise inclut le buffer, ce qui garantit la continuité de l'apprentissage.

---

## 9. Limitations liées au matériel

Le device est résolu automatiquement par `utils/device.resolve_device("auto")` : `cuda > mps > cpu`. Toutes les configs déclarent `device: auto` — aucune modification YAML nécessaire selon la machine.

### Apple Silicon (MPS)

- `resolve_device("auto")` retourne `"mps"` sur macOS Apple Silicon (SB3 v2.3.x ne le fait pas nativement — le wrapper corrige ce comportement).
- Vitesse mesurée : ~2 600 steps/min avec 12 workers et `gradient_steps=12`.
- Le déterminisme parfait n'est pas garanti sur MPS (opérations atomiques GPU).
- Pour des résultats parfaitement reproductibles : `device: cpu` dans le YAML (plus lent mais déterministe).
- `MUJOCO_GL` auto-défini à `cgl` par `factory.py`.

### GPU NVIDIA (CUDA) — Linux / WSL2 / Windows

- `resolve_device("auto")` retourne `"cuda"` si disponible.
- Vitesse mesurée : ~3 500 steps/min sur RTX 4070.
- Le déterminisme CUDA peut varier selon la version de cuDNN.
- `MUJOCO_GL` auto-défini à `egl` (Linux/WSL2) ou `wgl` (Windows).

### CPU uniquement

- Fonctionne sur toutes les plateformes sans GPU.
- Plus lent (~800–1 200 steps/min).
- Déterminisme parfait.
- Forcer avec `device: cpu` dans le YAML ou via `PYTHONHASHSEED=0`.

### RAM

- 12 workers SAC : ~10 Go RAM au total (3.3 Go process principal + 5.5 Go workers).
- Requis minimum : 16 Go RAM.
- Recommandé : 32 Go pour confort + MLflow UI simultané.

---

## 10. Artefacts lourds — stratégie Git

Les artefacts suivants **ne sont pas versionnés dans Git** (listés dans `.gitignore`) :

| Artefact | Taille typique | Stratégie |
|---|---|---|
| `checkpoints/` | ~50 Mo/checkpoint | Régénérer avec `make train-sac` |
| `outputs/` | ~100 Mo/run | Régénérer avec `make eval-*` |
| `mlruns/` | Variable | Régénérer avec le training |
| `logs/` | ~50 Mo | Régénérer avec TensorBoard |
| `.venv/` | ~3 Go | Régénérer avec `bash scripts/setup_uv.sh` |
| `external/` | ~500 Mo | Régénéré par `setup_uv.sh` (commits figés) |

**Pour partager un checkpoint :** utiliser un service externe (Google Drive, HuggingFace Hub, etc.) et documenter le lien dans `docs/results.md`.

**Pour partager les courbes :** commiter les PNG générés dans `outputs/plots/` (taille raisonnable, ~500 Ko chacun).
