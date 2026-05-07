# Reproductibilité, rendu et collaboration

Ce document regroupe les éléments de reproductibilité, la checklist de rendu et les notes de collaboration Git.


---

## Reproductibilité — Guide complet

---

### 1. Environnement testé

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

### 2. Installation reproductible

#### Étape 1 — Cloner le repo

```bash
git clone https://github.com/YiZeems/robocasa-project-B.git
cd robocasa-project-B
## Optionnel : se placer sur un commit spécifique pour la reproductibilité parfaite
## git checkout <commit_sha>
```

#### Étape 2 — Installation complète

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

#### Étape 3 — Vérification

```bash
pytest tests/test_platform_smoke.py -v   # 11 tests cross-plateforme, < 5 s, sans GPU
make check    # compile + vérification des imports
make sanity   # smoke test 20 steps reset/step
```

---

### 3. Seeds et déterminisme

#### Seeds utilisés

| Seed | Usage | Valeur par défaut |
|---|---|---|
| `train.seed` | Seed d'entraînement (policy init, replay buffer) | 0 |
| `eval.validation_seed` | Seed pour les évaluations périodiques pendant le training | 10 000 |
| `eval.test_seed` | Seed pour l'évaluation finale (split test) | 20 000 |
| `base_seed` (eval_video) | Seed de base pour les N épisodes scorés | 0 |

**Pourquoi des seeds séparés ?** Le split validation (seed=10000) et le split test (seed=20000) garantissent que l'agent n'a jamais vu les configurations initiales du split test pendant le training ou la sélection de checkpoint.

#### Niveau de déterminisme

Sur CPU, les runs sont **entièrement reproductibles** avec le même seed. Sur GPU (CUDA), des variations minimes peuvent apparaître dues aux opérations non-déterministes de cuDNN. Sur MPS (Apple Silicon), le déterminisme est intermédiaire.

Pour un déterminisme maximum :

```bash
PYTHONHASHSEED=0 uv run python -m robocasa_telecom.train \
  --config configs/train/open_single_door_sac.yaml \
  --seed 0 --device cpu
```

---

### 4. Reproduire un run complet

#### SAC principal (3M steps, seed=0)

```bash
## 1. Installation
bash scripts/setup_uv.sh

## 2. Entraînement
uv run python -m robocasa_telecom.train \
  --config configs/train/open_single_door_sac.yaml \
  --seed 0

## 3. Évaluation sur le split test
make eval-test \
  CONFIG=configs/train/open_single_door_sac.yaml \
  CHECKPOINT=checkpoints/<run_id>/best_model.zip \
  EPISODES=50

## 4. Génération vidéo
make eval-video \
  CONFIG=configs/train/open_single_door_sac.yaml \
  CHECKPOINT=checkpoints/<run_id>/best_model.zip \
  EPISODES=20 SEED=0

## 5. Courbes
make plot PLOT_RUNS="outputs/OpenCabinet_SAC_seed0_*/"

## 6. MLflow UI
uv run mlflow ui --backend-store-uri ./mlruns
```

#### PPO baseline (5M steps, seed=0)

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

### 5. Reproduire les courbes

```bash
## Courbes d'un seul run
uv run python scripts/plot_training.py \
  --run outputs/OpenCabinet_SAC_seed0_<timestamp>/ \
  --out outputs/plots/ \
  --smooth 3

## Comparaison SAC vs PPO
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

### 6. Reproduire la vidéo

```bash
make eval-video \
  CONFIG=configs/train/open_single_door_sac.yaml \
  CHECKPOINT=checkpoints/<run_id>/best_model.zip \
  EPISODES=20 SEED=0 VIDEO_OUT=outputs/eval/videos/
```

La vidéo est reproductible : avec le même checkpoint, le même seed et la même config, `eval_video.py` produit exactement les mêmes trajectoires (politique déterministe + seed fixé).

---

### 7. Fichiers de reproductibilité par run

Pour chaque run, les fichiers suivants permettent de reproduire exactement les résultats :

| Fichier | Contenu | Usage |
|---|---|---|
| `outputs/<run_id>/resolved_train_config.yaml` | Config exacte utilisée (tous overrides résolus) | Reproduire avec la même config |
| `outputs/<run_id>/train_summary.json` | Résumé complet : seed, steps, meilleur checkpoint, métriques | Vérifier les résultats |
| `outputs/<run_id>/validation_curve.csv` | Toutes les métriques validation step par step | Reproduire les courbes |
| `checkpoints/<run_id>/sac_<step>_steps.json` | Métadonnées du checkpoint (num_timesteps, run_id) | Reprendre depuis ce checkpoint |
| `uv.lock` | Versions exactes de toutes les dépendances | Reproduire l'environnement |

---

### 8. Reprendre un run interrompu

```bash
## Reprise automatique (détecte le dernier run incomplet compatible)
uv run python -m robocasa_telecom.train \
  --config configs/train/open_single_door_sac.yaml \
  --seed 0

## Reprise explicite depuis un checkpoint
uv run python -m robocasa_telecom.train \
  --config configs/train/open_single_door_sac.yaml \
  --seed 0 \
  --resume-from checkpoints/<run_id>/sac_1000000_steps.zip

## Forcer un départ depuis zéro (ignore l'auto-resume)
uv run python -m robocasa_telecom.train \
  --config configs/train/open_single_door_sac.yaml \
  --seed 0 --no-auto-resume
```

Pour SAC, le replay buffer est sauvegardé avec chaque checkpoint périodique (`sac_<step>_steps_replay_buffer.pkl`). La reprise inclut le buffer, ce qui garantit la continuité de l'apprentissage.

---

### 9. Limitations liées au matériel

Le device est résolu automatiquement par `utils/device.resolve_device("auto")` : `cuda > mps > cpu`. Toutes les configs déclarent `device: auto` — aucune modification YAML nécessaire selon la machine.

#### Apple Silicon (MPS)

- `resolve_device("auto")` retourne `"mps"` sur macOS Apple Silicon (SB3 v2.3.x ne le fait pas nativement — le wrapper corrige ce comportement).
- Vitesse mesurée : ~2 600 steps/min avec 12 workers et `gradient_steps=12`.
- Le déterminisme parfait n'est pas garanti sur MPS (opérations atomiques GPU).
- Pour des résultats parfaitement reproductibles : `device: cpu` dans le YAML (plus lent mais déterministe).
- `MUJOCO_GL` auto-défini à `cgl` par `factory.py`.

#### GPU NVIDIA (CUDA) — Linux / WSL2 / Windows

- `resolve_device("auto")` retourne `"cuda"` si disponible.
- Vitesse mesurée : ~3 500 steps/min sur RTX 4070.
- Le déterminisme CUDA peut varier selon la version de cuDNN.
- `MUJOCO_GL` auto-défini à `egl` (Linux/WSL2) ou `wgl` (Windows).

#### CPU uniquement

- Fonctionne sur toutes les plateformes sans GPU.
- Plus lent (~800–1 200 steps/min).
- Déterminisme parfait.
- Forcer avec `device: cpu` dans le YAML ou via `PYTHONHASHSEED=0`.

#### RAM

- 12 workers SAC : ~10 Go RAM au total (3.3 Go process principal + 5.5 Go workers).
- Requis minimum : 16 Go RAM.
- Recommandé : 32 Go pour confort + MLflow UI simultané.

---

### 10. Artefacts lourds — stratégie Git

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


---

## Checklist de rendu — IA705, Telecom Paris

> Deadline : **jeudi 7 mai 2026 à 23h59**
> Cocher chaque item avant le rendu.

---

### Vérifications techniques (code)

- [ ] `make check` passe sans erreur
- [ ] `make sanity` passe sans erreur (20 steps reset/step)
- [ ] Smoke test : `uv run python -m robocasa_telecom.train --config configs/train/open_single_door_sac_debug.yaml --seed 0 --total-timesteps 10 --no-auto-resume`
- [ ] Pas de fichier sensible ou chemin absolu personnel dans Git
- [ ] `.gitignore` couvre `checkpoints/`, `outputs/`, `mlruns/`, `logs/`, `.venv/`
- [ ] Aucune clé API, token ou secret dans le code

---

### Entraînement

- [ ] SAC debug (300k) terminé — `outputs/OpenCabinet_SAC_seed0_*/` existe
- [ ] SAC principal (3M) terminé — `checkpoints/<run_id>/best_model.zip` existe
- [ ] PPO baseline (5M) terminé — `checkpoints/<run_id_ppo>/best_model.zip` existe
- [ ] SAC tuned (2M) terminé (optionnel — si temps disponible)
- [ ] `train_summary.json` présent pour chaque run complété
- [ ] `validation_curve.csv` présent pour chaque run complété

---

### Évaluation

- [ ] `make eval-test` exécuté sur SAC principal `best_model.zip` — résultats dans `outputs/eval/`
- [ ] `make eval-test` exécuté sur PPO baseline `best_model.zip` — résultats dans `outputs/eval/`
- [ ] `test_success_rate` reporté dans `docs/results.md`
- [ ] Écart train/validation/test analysé et documenté

---

### Vidéos

- [ ] `make eval-video` exécuté sur SAC principal (20 épisodes, seed=0)
- [ ] `<run_id>_best_episode_*.mp4` existe et est lisible
- [ ] `<run_id>_worst_episode_*.mp4` existe (pour analyse des échecs)
- [ ] Vidéo décrite dans `docs/results.md` (comportement observé)
- [ ] Vidéo loggée dans MLflow (artefact `videos/`)

---

### Métriques et courbes

- [ ] `make plot` exécuté — `outputs/plots/` contient `summary.png`, `success_rate.png`, `door_angle.png`, `anti_hacking.png`
- [ ] Courbes insérées ou référencées dans `docs/results.md`
- [ ] Métriques anti-hacking vérifiées (`approach_frac < 0.5`, `stagnation < 100`)
- [ ] Tableau comparatif SAC vs PPO rempli dans `docs/results.md`
- [ ] Best checkpoint vs final checkpoint comparé

---

### MLflow

- [ ] `uv run mlflow ui --backend-store-uri ./mlruns` fonctionne
- [ ] Runs SAC et PPO visibles dans l'interface MLflow
- [ ] Métriques `val_success_rate`, `val_approach_frac_mean` présentes
- [ ] Screenshots ou export MLflow joint au rapport si nécessaire

---

### Documentation

- [ ] `README.md` complet et à jour
- [ ] `docs/project_report.md` rédigé et tous les `[À compléter]` remplacés
- [ ] `docs/results.md` rempli avec les valeurs réelles
- [ ] `docs/reward_shaping.md` présent
- [ ] `docs/metrics.md` présent
- [ ] `docs/video_generation.md` présent
- [ ] `docs/reproducibility.md` présent
- [ ] `docs/troubleshooting.md` présent

---

### Rapport académique

- [ ] Introduction : contexte, objectif, question de recherche
- [ ] Tâche : description OpenCabinet, robot PandaOmron, horizon, observation, action
- [ ] Formulation MDP : états, actions, reward, γ, succès/échec
- [ ] Méthodes : SAC (principal), PPO (baseline), justification des choix
- [ ] Reward shaping : formule complète, coefficients, mécanismes anti-hacking
- [ ] Protocole : n_envs, seeds, eval_freq, n_eval_episodes, split validation/test
- [ ] Résultats : tableaux + courbes + vidéos
- [ ] Discussion : SAC vs PPO, convergence, surentraînement
- [ ] Limites : seed unique, pas de sim-to-real, navigation exclue
- [ ] Perspectives : multi-seeds, curriculum, imitation learning, RGB
- [ ] Conclusion : réponse à la question de recherche
- [ ] Références : RoboCasa, SAC, PPO, SB3, RoboSuite, MuJoCo

---

### Git et GitHub

- [ ] Tous les commits pushés sur `main`
- [ ] Pas de fichier `mlruns/`, `checkpoints/`, `.venv/` dans Git
- [ ] Courbes PNG dans `outputs/plots/` committées (optionnel mais utile)
- [ ] `train_summary.json` des runs finaux committés (optionnel)
- [ ] README visible et bien rendu sur GitHub

---

### Ablations (optionnel — renforce le rapport)

- [ ] Run sans pénalité stagnation (`w_stagnation=0`) pour montrer l'effet
- [ ] Run avec reward naive (sans anti-hacking) pour montrer le hover hacking
- [ ] Comparaison `approach_frac` avant vs après le reward shaping

---

### Éléments encore à produire (mise à jour le 5 mai 2026)

| Élément | Statut | Action |
|---|---|---|
| SAC debug 300k | ✅ Terminé | — |
| SAC principal 3M | ⏳ En cours | Lancer `make train-sac SEED=0` |
| PPO baseline 5M | ⏳ À lancer | Lancer `make train-ppo-baseline SEED=0` |
| Eval test SAC | ⏳ Après run | `make eval-test CONFIG=... CHECKPOINT=...` |
| Eval test PPO | ⏳ Après run | `make eval-test CONFIG=... CHECKPOINT=...` |
| Vidéos | ⏳ Après run | `make eval-video CONFIG=... CHECKPOINT=...` |
| Courbes PNG | ⏳ Après run | `make plot PLOT_RUNS=...` |
| docs/results.md rempli | ⏳ Après runs | Remplir les placeholders |
| project_report.md finalisé | ⏳ Mercredi 6 mai | Rédaction |

---

### Recommandations finales

1. **Utiliser `best_model.zip`**, jamais `final_model.zip`, pour les résultats du rapport.
2. **Vérifier `approach_frac`** avant de conclure que le reward shaping fonctionne.
3. **Comparer best vs final** pour documenter le surentraînement.
4. **Citer les métriques anti-hacking** dans le rapport — c'est un apport méthodologique fort.
5. **Inclure une vidéo** : les examinateurs apprécient voir le comportement réel.


---

## Collaboration Git (équipe de 4)

### 1) Branches de travail

Branches recommandées et déjà préparées:
- `main`: branche stable, démontrable, exécutable.
- `develop`: intégration continue d'équipe.
- `feature/member1-env-foundation`
- `feature/member2-training-pipeline`
- `feature/member3-eval-visualization`
- `feature/member4-slurm-repro`

### 2) Répartition conseillée des responsabilités

- Member 1 (`feature/member1-env-foundation`):
  - `robocasa_telecom/envs/`
  - configs environnement
  - robustesse setup et compatibilité tâches

- Member 2 (`feature/member2-training-pipeline`):
  - `robocasa_telecom/rl/train.py`
  - tuning hyperparamètres
  - export des métriques d'entraînement

- Member 3 (`feature/member3-eval-visualization`):
  - `robocasa_telecom/rl/evaluate.py`
  - `scripts/visualize_env.py`
  - scripts d'analyse et graphes

- Member 4 (`feature/member4-slurm-repro`):
  - `scripts/slurm/`
  - wrappers run local/cluster
  - reproductibilité, tests de job array

### 3) Workflow recommandé

1. Sync local:
   - `git checkout develop`
   - `git pull origin develop`

2. Travail feature:
   - `git checkout feature/memberX-...`
   - commits atomiques et ciblés.

3. Pull Request:
   - PR de `feature/*` vers `develop`.
   - review croisée (au moins 1 reviewer).

4. Promotion stable:
   - merge `develop` vers `main` seulement après validation exécutable.

### 4) Convention de commit

Format conseillé:
- `<scope>: <action concise>`

Scopes utilisés dans ce dépôt:
- `env`, `rl`, `tools`, `utils`, `scripts`, `slurm`, `docs`, `tests`

Exemples:
- `env: add robust OpenSingleDoor alias handling`
- `rl: export training curve from monitor csv`
- `slurm: map array index to explicit seed list`
- `docs: add file-by-file technical reference`

### 5) Checklist PR

Avant merge vers `develop`:
- code exécutable en local,
- script shell valide (`bash -n`),
- modules Python compilent (`python -m compileall`),
- test config minimal passe (`tests/test_config_loading.py`),
- doc mise à jour si interface ou comportement modifié.

### 6) Gestion des conflits

- Toujours rebase ou merge `develop` avant gros commit final.
- Ne jamais réécrire l'historique de `main`.
- En cas de conflit sur un fichier partagé (`README`, config train), faire une mini-réunion d'alignement avant résolution.
