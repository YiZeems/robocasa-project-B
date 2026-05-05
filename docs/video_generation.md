# Génération vidéo — Guide complet

> Implémentation : `robocasa_telecom/rl/eval_video.py`  
> Commande Make : `make eval-video`

---

## 1. Pourquoi générer des vidéos ?

Les vidéos servent à :

1. **Comprendre le comportement de l'agent** : les métriques quantitatives (success_rate, door_angle) ne révèlent pas *comment* l'agent ouvre la porte, ni s'il adopte un comportement étrange (oscillation, approche par le dessus, etc.).
2. **Vérifier l'absence de reward hacking** : une vidéo montre immédiatement si l'agent reste immobile devant la poignée ou oscille la porte.
3. **Communication** : les vidéos sont plus convaincantes qu'un tableau de métriques pour présenter les résultats au professeur.
4. **Débogage** : comparer la vidéo du meilleur épisode avec celle du pire épisode révèle les limites de la politique.

---

## 2. Contrainte principale : SubprocVecEnv

Pendant l'entraînement, **il est impossible de générer des vidéos depuis les workers parallèles**. Les workers `SubprocVecEnv` s'exécutent dans des processus séparés (fork/spawn), et le contexte OpenGL/MuJoCo ne peut pas être partagé entre processus.

Options théoriques :

| Option | Description | Problème |
|---|---|---|
| A — Pendant l'entraînement | Rendre dans le worker | Impossible avec SubprocVecEnv |
| B — Post-training | Re-jouer avec un env dédié | **Solution retenue** |
| C — Env worker unique | Entraîner avec 1 seul worker pour avoir le rendu | Beaucoup plus lent |

**Option B retenue** : génération post-training via `eval_video.py` avec une approche two-pass.

---

## 3. Approche two-pass

### Pass 1 — Scoring sans rendu (rapide)

- N épisodes (par défaut : 20) joués avec l'environnement **sans rendu**.
- Toutes les métriques sont collectées à chaque step (θ, d_ee, reward components).
- Chaque épisode reçoit un **score composite** anti-hacking.
- Durée : ~30–120 secondes selon N et le checkpoint.

### Pass 2 — Rendu du meilleur épisode (reproductible)

- Le meilleur et le pire épisodes sont **re-joués** avec **le même seed** dans un environnement avec rendu offscreen activé.
- Reproductible : même seed + même politique déterministe = même trajectoire.
- Les frames sont assemblées en grille 2×2 (4 caméras bras) et exportées en MP4.

---

## 4. Critère de sélection du meilleur épisode

```
score = 1000 × success
       + 100  × door_angle_final   [normalised 0-1]
       + 10   × door_angle_max     [high-watermark]
       + 1    × episode_return     [tiebreaker]
       - 0.1  × stagnation_steps   [pénalise hover hacking]
       - 0.01 × episode_length     [préfère les succès rapides]
```

**Priorité explicite :**
1. Succès (pondéré 1000 → domine tout)
2. Angle final (pondéré 100 → parmi les succès, préfère la plus grande ouverture)
3. Angle max (pondéré 10 → high-watermark, anti-oscillation)
4. Return (pondéré 1 → tiebreaker seulement)
5. Pénalité stagnation (−0.1 → discrimine les succès "hover-hacky")
6. Pénalité longueur (−0.01 → préfère les succès rapides)

Ce scoring est **résistant au reward hacking** : un agent qui réussit vite et proprement a un score bien supérieur à un agent qui réussit en oscillant longtemps.

---

## 5. Commandes

### Commande Make (recommandée)

```bash
make eval-video \
  CONFIG=configs/train/open_single_door_sac.yaml \
  CHECKPOINT=checkpoints/<run_id>/best_model.zip \
  EPISODES=20 \
  SEED=0
```

Variables Makefile disponibles :

| Variable | Défaut | Description |
|---|---|---|
| `EPISODES` | 20 | Nombre d'épisodes pour le scoring (pass 1) |
| `SEED` | 0 | Seed de base |
| `VIDEO_OUT` | `outputs/eval/videos` | Dossier de sortie |
| `VIDEO_FPS` | 20 | Images par seconde du MP4 |

### CLI complet

```bash
uv run python -m robocasa_telecom.rl.eval_video \
  --config configs/train/open_single_door_sac.yaml \
  --checkpoint checkpoints/<run_id>/best_model.zip \
  --episodes 20 \
  --seed 0 \
  --out outputs/eval/videos/ \
  --fps 20
```

### Générer sans vidéo du pire épisode

```bash
uv run python -m robocasa_telecom.rl.eval_video \
  --config configs/train/open_single_door_sac.yaml \
  --checkpoint checkpoints/<run_id>/best_model.zip \
  --episodes 20 --seed 0 \
  --no-worst
```

### Attacher à un run MLflow existant

```bash
uv run python -m robocasa_telecom.rl.eval_video \
  --config configs/train/open_single_door_sac.yaml \
  --checkpoint checkpoints/<run_id>/best_model.zip \
  --episodes 20 --seed 0 \
  --mlflow-run-id <mlflow_run_id>
```

---

## 6. Fichiers générés

```text
outputs/eval/videos/
  <run_id>_best_episode_<timestamp>.mp4
      ← Vidéo du meilleur épisode (grille 4 caméras bras)
  <run_id>_worst_episode_<timestamp>.mp4
      ← Vidéo du pire épisode (optionnel, pour analyse des échecs)
  <run_id>_best_episode_metadata_<timestamp>.json
      ← Toutes les métadonnées du meilleur épisode
  <run_id>_video_selection_debug_<timestamp>.csv
      ← Une ligne par épisode avec score + métriques (pour audit)
```

### Structure du JSON de métadonnées

```json
{
  "run_id": "OpenCabinet_SAC_seed0_20260505_120000",
  "checkpoint_path": "checkpoints/.../best_model.zip",
  "algorithm": "SAC",
  "n_episodes_scored": 20,
  "base_seed": 0,
  "deterministic": true,
  "cameras": ["robot0_eye_in_hand", "robot0_agentview_center", ...],
  "best": {
    "episode_id": 7,
    "seed_used": 7,
    "success": true,
    "episode_return": 312.4,
    "door_angle_final": 0.943,
    "door_angle_max": 0.943,
    "stagnation_steps": 0,
    "time_to_success": 142,
    "score": 1194.3,
    "reason_selected": "max(1000*success + ...)",
    "video_path": "outputs/eval/videos/...best_episode.mp4"
  },
  "aggregate": {
    "success_rate": 0.65,
    "return_mean": 198.2,
    "door_angle_final_mean": 0.71
  }
}
```

---

## 7. Logging MLflow

Les métriques et les vidéos sont automatiquement loggées dans MLflow si un run actif existe ou si `--mlflow-run-id` est fourni.

**Métriques loggées :**

| Métrique MLflow | Description |
|---|---|
| `eval_video/success_rate` | Taux de succès sur les N épisodes scorés |
| `eval_video/return_mean` | Return moyen |
| `eval_video/door_angle_final_mean` | Angle de porte final moyen |
| `eval_video/door_angle_max_mean` | High-watermark moyen |
| `eval_video/best_success` | Le meilleur épisode est-il un succès ? |
| `eval_video/best_door_angle_final` | Angle final du meilleur épisode |
| `eval_video/best_score` | Score du meilleur épisode |

**Artefacts loggés :**
- `videos/best_episode.mp4`
- `videos/worst_episode.mp4`
- `videos/best_episode_metadata.json`
- `videos/video_selection_debug.csv`

---

## 8. Vidéo en 4 vues

Le rendu utilise 4 caméras orientées sur le bras et la pince, assemblées en grille 2×2 (512×512 px) :

```
+------------------+------------------+
|  eye_in_hand     |  agentview_center|
+------------------+------------------+
|  frontview       |  sideview        |
+------------------+------------------+
```

Les caméras disponibles sont résolues automatiquement via `resolve_arm_video_cameras()` depuis la configuration d'environnement. Si une caméra n'est pas disponible, un frame noir est inséré.

---

## 9. Vérifier que la vidéo correspond au bon épisode

Pour vérifier que la vidéo correspond bien au meilleur épisode :

1. Ouvrir `video_selection_debug.csv`
2. Chercher la ligne avec `selected_as_best=1`
3. Vérifier que `seed_used`, `success`, `score`, `door_angle_final` correspondent aux valeurs dans le JSON de métadonnées
4. Re-jouer manuellement le même épisode :

```bash
# Test de reproductibilité : rejouer le seed du meilleur épisode
uv run python -c "
from robocasa_telecom.envs.factory import load_env_config, make_env_from_config
from stable_baselines3 import SAC
cfg = load_env_config('configs/env/open_single_door.yaml')
env = make_env_from_config(cfg, seed=7)  # seed_used du meilleur épisode
model = SAC.load('checkpoints/<run_id>/best_model.zip')
obs, _ = env.reset(seed=7)
print('Reproductibilité OK si comportement identique')
env.close()
"
```

---

## 10. Cas des 12 workers

Pendant l'entraînement avec 12 workers, chaque worker génère des épisodes avec des seeds différents. La vidéo post-training via `eval_video.py` utilise un **seul worker dédié** et rejoue les épisodes séquentiellement — aucun problème de rendu.

Le seed utilisé pour chaque épisode est : `base_seed + episode_id`. Pour N=20 épisodes et `base_seed=0`, les seeds sont 0, 1, 2, ..., 19.

---

## 11. Dépannage vidéo

| Symptôme | Cause probable | Solution |
|---|---|---|
| Vidéo noire | Rendu offscreen non initialisé | Vérifier MuJoCo + `has_offscreen_renderer: true` |
| Vidéo courte | `max_steps` trop petit | Augmenter via `--fps` ou modifier `eval_video.py` |
| Même seed mais trajectoire différente | Non-déterminisme de l'env | Vérifier que le même checkpoint est utilisé |
| Score 0 pour tous les épisodes | Problème d'extraction de θ | Vérifier `reward_components` dans `info` |
| Erreur `imageio` | Dépendance vidéo manquante | `uv run pip install imageio[ffmpeg]` |
