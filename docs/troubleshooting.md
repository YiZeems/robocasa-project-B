# Troubleshooting — Résolution des problèmes courants

---

## Installation et setup

### `.venv` manquant ou corrompu

**Symptôme :** `uv run` échoue avec "No virtual environment found".

```bash
# Solution
bash scripts/setup_uv.sh
```

Si le setup échoue à mi-parcours :

```bash
rm -rf .venv
bash scripts/setup_uv.sh
```

---

### Assets RoboCasa manquants

**Symptôme :** `FileNotFoundError` mentionnant `robocasa/models/assets/`.

```bash
# Solution : retélécharger les assets
DOWNLOAD_ASSETS=1 VERIFY_ASSETS=1 bash scripts/setup_uv.sh
```

Si les assets sont déjà sur le disque mais mal liés :

```bash
# Vérifier le lien symbolique
ls -la external/robocasa/robocasa/models/assets/
# Si cassé, relancer le setup
bash scripts/setup_uv.sh
```

---

### Warnings `mink`, `mimicgen`, `gym`

**Symptôme :**
```
WARNING: mimicgen environments not imported since mimicgen is not installed!
WARNING: mink environments not imported...
```

**Cause :** dépendances optionnelles non installées.

**Solution :** ces warnings sont **non bloquants** et attendus sur cette stack. Ignorer. Les fonctionnalités utilisées dans ce projet ne dépendent ni de mimicgen ni de mink.

---

## MLflow

### MLflow UI ne s'affiche pas / port occupé

**Symptôme :** `Address already in use` sur le port 5000.

```bash
# Utiliser un port alternatif
uv run mlflow ui --backend-store-uri ./mlruns --port 5001
# Puis ouvrir http://127.0.0.1:5001
```

Sur macOS, le port 5000 est souvent occupé par AirPlay Receiver :
- Désactiver dans Réglages Système → AirDrop et Handoff → AirPlay Receiver
- Ou simplement utiliser le port 5001

---

### Aucun run dans MLflow UI

**Symptôme :** l'interface MLflow s'ouvre mais affiche "No runs found".

**Causes probables :**
1. Le dossier `mlruns/` est vide (aucun run n'a été lancé)
2. Le `backend-store-uri` pointe au mauvais endroit

```bash
# Vérifier que mlruns existe et contient des données
ls mlruns/
# S'assurer d'être dans le bon répertoire
uv run mlflow ui --backend-store-uri ./mlruns
```

---

## Entraînement

### Reward élevée mais 0 % de succès (hover hacking)

**Symptôme :** `val_return_mean` croît mais `val_success_rate` reste à 0. `val_approach_frac_mean > 0.5`.

**Cause :** l'agent a appris à rester proche de la poignée sans ouvrir la porte.

**Diagnostic dans MLflow :**
- Ouvrir MLflow UI
- Chercher `val_approach_frac_mean` — si > 0.5, c'est confirmé
- Chercher `val_door_angle_max_mean` — si < 0.2, l'agent n'a jamais bien ouvert

**Solutions :**
1. Augmenter `w_stagnation` (de 0.05 à 0.1–0.2) dans `configs/env/open_single_door.yaml`
2. Réduire `w_approach` (de 0.05 à 0.01)
3. Vérifier que `AntiHackingReward` est bien utilisé (non la reward native RoboCasa)
4. Lancer un SAC debug plus long (500k au lieu de 300k)

---

### Run reprend en boucle infinie (auto-resume)

**Symptôme :** le run détecte un run précédent comme "incomplet" et le reprend, mais le checkpoint contient déjà 300k steps.

**Cause :** un run a crashé après `final_model.zip` mais avant `train_summary.json`. `_run_is_complete()` le considère incomplet.

```bash
# Solution : forcer un départ neuf
uv run python -m robocasa_telecom.train \
  --config configs/train/open_single_door_sac_debug.yaml \
  --seed 0 --no-auto-resume
```

---

### `VecMonitor.reset() got an unexpected keyword argument 'seed'`

**Symptôme :** erreur à la fin du training lors de l'évaluation finale.

**Cause :** ancien code utilisant `train_env` (VecMonitor) pour l'évaluation finale.

**Solution :** déjà corrigé dans `utils/metrics.py` — détecte `isinstance(env, VecEnv)` et utilise `env.seed(n); env.reset()`. Si l'erreur persiste, vérifier que vous utilisez la dernière version du code.

---

### `not enough values to unpack (expected 5, got 4)`

**Symptôme :** erreur lors du step dans l'évaluation.

**Cause :** `VecEnv.step()` retourne 4 valeurs (sans `truncated` séparé) dans l'ancienne API.

**Solution :** déjà corrigé. L'évaluation finale utilise un environnement single-worker avec l'API Gymnasium 0.29.1.

---

### Entraînement lent (< 1 000 steps/min)

**Causes possibles et solutions :**

| Cause | Diagnostic | Solution |
|---|---|---|
| `gradient_steps=12` est lent par design | Normal avec SAC + 12 workers | C'est intentionnel — trade-off sample efficiency |
| Device mal configuré | `device: cpu` au lieu de `mps`/`cuda` | Modifier dans le YAML ou `--device mps` |
| Trop de workers en RAM | Swap excessif | Réduire `n_envs` à 6 ou 4 |
| Disque SSD lent | Logs écrits fréquemment | Réduire `save_freq_steps` ou utiliser un SSD rapide |

**Vitesses de référence :**
- SAC 12 workers, MPS (M3 Pro) : ~2 600 steps/min
- SAC 12 workers, RTX 4070 : ~3 500 steps/min
- PPO 1 worker, CPU : ~800 steps/min

---

### OOM (Out of Memory) avec 12 workers

**Symptôme :** processus tué, `MemoryError`, ou swap excessif.

**Diagnostic :**

```bash
# Vérifier l'utilisation RAM
top -l 1 | grep -E "PhysMem|Swap"  # macOS
free -h                              # Linux
```

**Référence :** 12 workers SAC ≈ 10 Go RAM total (3.3 Go main + 5.5 Go workers).

**Solutions :**

```yaml
# Dans open_single_door_sac.yaml
n_envs: 6    # Réduire à 6 workers
# ou
n_envs: 4    # Réduire à 4 workers
```

Adapter aussi `gradient_steps` si `n_envs` change.

---

### `CUDA out of memory`

```bash
# Réduire la taille du batch
# Dans le YAML :
batch_size: 256  # au lieu de 512
# ou utiliser CPU
device: cpu
```

---

## Évaluation

### `best_model.zip` introuvable

**Symptôme :** `FileNotFoundError: checkpoints/<run_id>/best_model.zip`.

**Cause :** le run n'a pas encore terminé, ou le premier checkpoint n'a pas été sauvegardé (trop peu de steps).

**Solutions :**
1. Utiliser `final_model.zip` comme fallback temporaire
2. Vérifier que `eval.eval_freq > 0` dans le YAML (sinon la validation ne s'exécute jamais)
3. Vérifier que `eval.n_eval_episodes > 0`

---

### Success rate différent entre deux runs avec le même seed

**Causes possibles :**
1. GPU non-déterministe (CUDA/MPS) — utiliser `device: cpu` pour la reproductibilité parfaite
2. Version de RoboCasa différente — vérifier que `setup_uv.sh` utilise les mêmes commits
3. Seed de l'environnement non fixé lors de l'évaluation — vérifier `--seed` dans la commande eval

---

## Vidéo

### Vidéo vide (noire)

**Causes et solutions :**

1. `has_offscreen_renderer` non activé :
   - `eval_video.py` l'active automatiquement — vérifier que vous utilisez bien `eval_video.py` et non `evaluate.py` pour la vidéo

2. MuJoCo sans support d'affichage (serveur headless) :
   ```bash
   # Forcer le rendu EGL (headless)
   MUJOCO_GL=egl uv run python -m robocasa_telecom.rl.eval_video ...
   ```

3. Display manquant sur Linux :
   ```bash
   # Installer un display virtuel
   Xvfb :1 -screen 0 1280x1024x24 &
   DISPLAY=:1 uv run python -m robocasa_telecom.rl.eval_video ...
   ```

---

### Erreur `imageio` / `ffmpeg`

**Symptôme :** `ImageIOError: No reader found for format`.

```bash
# Réinstaller imageio avec support ffmpeg
uv run pip install imageio[ffmpeg]
# ou
uv run pip install imageio imageio-ffmpeg
```

---

### Vidéo courte ou trajectoire différente de celle attendue

**Symptôme :** la vidéo dure 2 secondes au lieu de 10.

**Cause :** l'épisode se termine très vite (succès rapide ou timeout à 0 steps).

**Vérifier :** `max_steps` dans `_render_episode()` — par défaut 600 steps (> 500 de l'épisode), donc pas un problème.

**Si la trajectoire semble différente entre pass 1 et pass 2 :**
- Vérifier que le même seed est utilisé
- Vérifier que la politique est déterministe (`deterministic=True`)
- Vérifier que le même checkpoint est utilisé

---

## Environnement RoboCasa

### Shape d'observation instable entre resets

**Symptôme :** `ValueError: observation space shape changed between resets`.

**Cause :** `GymWrapper` RoboSuite retourne parfois des shapes différentes.

**Solution :** déjà géré automatiquement — `make_env_from_config()` détecte ce cas et bascule sur `RawRoboCasaAdapter` qui flattèn les observations. Ce comportement est **attendu** et n'indique pas d'erreur.

---

### Collisions excessives / robot qui tremble

**Symptôme :** le robot vibre ou entre en collision avec le placard.

**Causes et solutions :**
1. `w_action_reg` trop faible → augmenter à 0.05 ou 0.1
2. `learning_rate` trop élevé → réduire à 1e-4
3. `control_freq` trop élevé → réduire à 5 Hz pour des actions plus stables

---

### Résultats différents entre seeds

C'est **attendu et normal** — la variance entre seeds est une propriété fondamentale des algorithmes RL. Pour la caractériser :

```bash
# Lancer 3 seeds
make train-sac SEED=0
make train-sac SEED=1
make train-sac SEED=2

# Comparer
uv run python scripts/plot_training.py \
  --run outputs/OpenCabinet_SAC_seed0_*/ \
        outputs/OpenCabinet_SAC_seed1_*/ \
        outputs/OpenCabinet_SAC_seed2_*/ \
  --label "seed0" "seed1" "seed2" \
  --out outputs/plots/multi_seed/
```
