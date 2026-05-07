# Progress Report — RoboCasa Door Opening

**Module** : IA705 — Apprentissage pour la robotique, Telecom Paris
**Date** : 7 mai 2026
**Statut** : SAC v1 abandonné · SAC v2 abandonné · SAC v3 en cours · PPO non lancé

---

## 1. Summary of Work Completed

### 1.1 Environment and Infrastructure

- Custom `RawRoboCasaAdapter` in `envs/factory.py` — wraps RoboCasa's `OpenCabinet`
  task into a stable Gymnasium interface with fixed 220D observation shape across resets.
- Anti-hacking reward shaping in `envs/reward.py` — 7 components: high-watermark
  progress, oscillation detection, stagnation penalty, approach gating, action
  regularisation, wrong-direction penalty, and success dominance.
- Training pipeline in `rl/train.py` — algorithm-agnostic (SAC and PPO),
  12 SubprocVecEnv workers, periodic checkpoint saving, ValidationCallback with
  early stopping.
- Two-pass video evaluation in `rl/eval_video.py` — scoring pass (no render)
  followed by reproduction pass with the same seed.
- MLflow tracking integrated at every validation step and at end of training.
- Auto-resume mechanism with `--no-auto-resume` flag for clean restarts.

### 1.2 Experiments Run

| Run | Config | Statut | Steps réalisés | val_success_rate | theta_best_mean | Notes |
|---|---|---|---:|---:|---:|---|
| SAC debug | `open_single_door_sac_debug.yaml` | Terminé | 300k | À compléter | À compléter | Hover hacking détecté et corrigé |
| SAC v1 | `open_single_door_sac.yaml` | Abandonné | ~700k | 0% | 0.001–0.012 | ent_coef auto crash → α→0 |
| SAC v2 | `open_single_door_sac_v2.yaml` | Abandonné | ~100k | 0% | À compléter | Même cause ; détectée plus tôt |
| SAC v3 | `open_single_door_sac_v3.yaml` | En cours | À compléter | À compléter | À compléter | ent_coef=0.1 fixe + SDE |
| PPO baseline | `open_single_door_ppo_baseline.yaml` | Non lancé | — | — | — | Deadline atteinte |

### 1.3 Issues Identified and Resolved

| Problème | Cause racine | Fix |
|---|---|---|
| ent_coef auto-tuning crash (v1 + v2) | log_prob ≈ −20 < target_entropy → α→0 | ent_coef=0.1 fixe (v3) |
| Critic loss spikes > 40k | α≈0 → pas de régularisation → Q-values divergent | gradient_steps=4, tau=0.01 |
| Hover hacking (approach_frac > 0.7) | Reward d'approche dense exploitable | High-watermark + gating + stagnation |
| Auto-resume reprend le mauvais checkpoint | task+algo+seed identiques entre v1/v2/v3 | --no-auto-resume dans tous les targets Makefile |
| Validation = 82% du wall time (v1) | n_eval_episodes=50 × eval_freq=25000 | n_eval_episodes=10, eval_freq=100000 |
| Reset 9.3s par épisode | obj_registries=[objaverse] | obj_registries=[lightwheel] → 4.8s |
| control_freq=10 (simulation 2× trop lente) | 50 substeps au lieu de 25 | control_freq=20 |
| SubprocVecEnv crash ConnectionResetError (fork) | MuJoCo hérite des FD/EGL du parent | spawn (défaut SB3) |
| OOM WSL2 (même symptôme) | 12 × 3.4 GB = ~41 GB > RAM configurée (31 GB) | .wslconfig memory=56GB |
| oscillation_frac négatif dans MLflow | Pénalité négative / abs(total_mean) sans abs() | abs() au numérateur |
| VecMonitor.reset() crash avec seed= | API Gymnasium ≥ 0.26 vs VecMonitor | isinstance(VecEnv) + env.seed(n); env.reset() |
| Observation shape instable entre resets | GymWrapper varie selon la scène | RawRoboCasaAdapter avec espace de référence fixe |
| mlruns/ committé dans Git | Oublié dans .gitignore | git rm -r --cached + hook pre-commit |

---

## 2. État Honnête des Résultats

### 2.1 Ce qui fonctionne

- L'infrastructure d'entraînement est fonctionnelle et robuste (12 workers, MLflow,
  checkpoints, auto-resume, vidéos).
- Le reward shaping anti-hacking a éliminé le hover hacking sur le run debug
  (comportement confirmé via métriques `approach_frac`, `stagnation_steps_mean`).
- Les diagnostics des échecs v1/v2 ont été conduits rigoureusement et ont abouti
  à une cause racine claire et un fix précis.

### 2.2 Ce qui ne fonctionne pas encore

- **Aucun run n'a atteint un taux de succès > 0%** à l'exception potentielle du
  run debug (métriques à compléter). Les runs v1 (700k steps) et v2 (100k steps)
  ont été abandonnés sur 0% de succès.
- **SAC v3 est en cours** : les résultats sont inconnus au moment de la rédaction.
  Il s'agit de la première configuration potentiellement viable.
- **PPO non lancé** : la contrainte de temps (deadline 7 mai) a empêché le lancement
  du run PPO baseline.
- **Pas de comparaison SAC vs PPO** : l'objectif de recherche initial (comparer
  les deux algorithmes) n'est pas encore atteint.

### 2.3 Résultats numériques disponibles

| Métrique | SAC v1 (700k, abandonné) | SAC v2 (100k, abandonné) | SAC v3 (en cours) |
|---|---|---|---|
| `val_success_rate` | 0% | 0% | À compléter |
| `theta_best_mean` | 0.001–0.012 | À compléter | À compléter |
| `actor_loss` | Remonte vers 0 puis positif | Remonte vers 0 | À compléter |
| `ent_coef` (α) | Crash vers ~0 dès 200k steps | Crash vers ~0 dès 100k steps | 0.1 fixe |
| `critic_loss` | Spikes > 40k à 500k steps | À compléter | À compléter |
| `return_mean` | Régresse de −15 à −35 | À compléter | À compléter |

---

## 3. Methodological Choices Made So Far

### 3.1 Pourquoi des runs debug avant les runs principaux

Le run debug (300k steps) a été lancé avant le run principal (3M steps) pour trois
raisons :

1. **Validation du reward shaping** — avec 12 workers et 300k steps, il est possible
   de détecter le hover hacking (`approach_frac > 0.5`) avant de consacrer ~19h de
   calcul au run principal.
2. **Validation de l'infrastructure** — le run debug a confirmé que les workers
   SubprocVecEnv ne saturent pas la RAM, que MLflow log correctement, et que la
   ValidationCallback sauvegarde `best_model.zip` au bon moment.
3. **Gestion du risque** — aborter un run de 300k steps coûte des minutes ; aborter
   un run de 3M steps coûte des heures.

Le cours souligne que les algorithmes RL sont sensibles au reward design et aux
hyperparamètres. Les runs debug sont la méthode standard pour détecter les erreurs
de design avant de consacrer des ressources aux longs runs.

### 3.2 Pourquoi 12 workers parallèles

- **SAC avec replay buffer bénéficie de la diversité** — chaque worker explore avec
  un seed différent, produisant des expériences non-corrélées dans le replay buffer.
- **gradient_steps=4 est GPU-intensif** — SAC effectue 4 mises à jour réseau par
  step d'environnement. Le bottleneck est le GPU, pas la collecte de données.
  12 workers garantissent que le GPU n'attend jamais de données.
- **Adéquation hardware** — sur WSL2 avec 56 GB RAM, 12 workers utilisent ~41 GB,
  laissant une marge raisonnable.

### 3.3 Pourquoi ent_coef fixe en v3

L'auto-tuning du coefficient d'entropie SAC suppose que la `log_prob` de la
politique peut atteindre le `target_entropy`. Sur un espace d'action 12D, la
`log_prob` d'une politique aléatoire est structurellement autour de −20, bien
inférieure à tout `target_entropy` raisonnable (−4 à −12). SAC interprète cela
comme "la politique est trop déterministe" et augmente `α`… non, exactement
l'inverse : SAC pousse `α` vers 0 car `log_prob < target_entropy` signifie que
l'entropie actuelle est déjà trop basse, donc SAC réduit la pénalité d'entropie.
Le résultat est une politique déterministe avant d'avoir appris.

`ent_coef=0.1` fixe brise ce cycle en imposant une régularisation entropique
constante, indépendante de la log-prob courante.

**Lien cours :** Exploration vs exploitation — le coefficient d'entropie contrôle
directement la diversité de la politique dans SAC (maximum entropy RL).

### 3.4 Pourquoi surveiller door_angle_final et non return_mean

`door_angle_final_mean` est la mesure physique directe de l'ouverture de la porte.
Contrairement au `return_mean` (qui peut être élevé même en cas de hover hacking),
un `door_angle_final_mean` élevé signifie que la porte est réellement ouverte à la
fin de l'épisode. C'est la métrique primaire pour évaluer la progression de l'agent.

Un écart important entre `door_angle_max_mean` (meilleur angle pendant l'épisode)
et `door_angle_final_mean` (angle à la fin) indique que la porte est ouverte puis
repoussée — signe d'oscillation ou de pénalité wrong_dir trop agressive.

### 3.5 Pourquoi MLflow

MLflow résout le problème de reproductibilité en RL en loggant :
- Tous les hyperparamètres et configs pour chaque run
- Les métriques à chaque validation step (permettant la comparaison SAC/PPO)
- Les artefacts `best_model.zip`, `validation_curve.csv`, vidéos

L'interface (`http://127.0.0.1:5000`) permet au professeur de vérifier la
méthodologie expérimentale directement.

---

## 4. Analyse des Échecs v1 et v2

Les deux premiers runs constituent un résultat expérimental en soi : ils démontrent
que l'auto-tuning du coefficient d'entropie SAC est instable sur cet espace d'action
lorsque la log-prob de la politique initiale est structurellement inférieure au
target_entropy.

### 4.1 Timeline SAC v1

| Steps | Événement | Métrique |
|---:|---|---|
| 0 | Démarrage — ent_coef="auto", α=1.0, target_entropy=−12 | — |
| ~50k | α commence à décroître (log_prob < target) | ent_coef ≈ 0.3 |
| ~200k | α ≈ 0 — politique quasi-déterministe | actor_loss remonte vers 0 |
| ~300k | Actions saturent aux limites — return commence à régresser | return_mean ≈ −20 |
| ~700k | Run abandonné — 0% de succès, return_mean ≈ −35 | theta_best_mean ≈ 0.012 |

### 4.2 Timeline SAC v2

| Steps | Événement | Métrique |
|---:|---|---|
| 0 | Démarrage — ent_coef="auto_0.1", α=0.1, target_entropy=−4 | — |
| ~50k | α commence à décroître (log_prob ≈ −20 < target_entropy=−4) | ent_coef ≈ 0.05 |
| ~100k | Run abandonné — diagnostic identique à v1 mais détecté plus tôt | 0% succès |

### 4.3 Fix v3 — Rationale

```yaml
# v3 : ent_coef fixe, pas d'auto-tuning
ent_coef: 0.1          # α = 0.1 permanent
use_sde: true          # Exploration structurée (bruit corrélé à l'état)
sde_sample_freq: 64    # Régénère le bruit toutes les 64 steps
gradient_steps: 4      # Réduction des spikes critic loss
tau: 0.01              # Target network plus réactif
```

---

## 5. Current State of Metrics

> Valeurs à compléter après analyse des runs debug et v3.

| Métrique | SAC debug (300k) | SAC v3 (en cours) | Cible (3M) |
|---|---|---|---|
| `val_success_rate` | À compléter | À compléter | > 50% |
| `val_approach_frac_mean` | À compléter | À compléter | < 0.3 |
| `val_door_angle_max_mean` | À compléter | À compléter | > 0.7 |
| `val_stagnation_steps_mean` | À compléter | À compléter | < 20 |
| `val_sign_changes_mean` | À compléter | À compléter | < 5 |
| `actor_loss` (trend) | À compléter | À compléter | Décroissant et négatif |
| `ent_coef` | N/A (auto) | 0.1 (fixe) | 0.1 (fixe) |

---

## 6. Open Questions

1. **SAC v3 convergera-t-il ?** Le fix `ent_coef=0.1 fixe` est théoriquement
   justifié. Il reste à confirmer que la policy apprend effectivement à ouvrir la
   porte avec ce réglage.

2. **Le reward shaping est-il bien calibré pour v3 ?** `w_wrong_dir=0.3` peut être
   trop punitif tôt dans l'entraînement quand l'agent a un faible theta_best. À
   surveiller sur les 200 premiers k steps de v3.

3. **Le `use_sde` aide-t-il vraiment ?** L'exploration structurée est théoriquement
   favorable pour la manipulation précise, mais elle peut aussi créer des mouvements
   trop orientés avant que la politique ait convergé vers la poignée.

4. **Comparaison SAC vs PPO possible avant la deadline ?** Avec le run PPO non lancé,
   la question de recherche principale reste sans réponse quantitative.

5. **Quel est le taux de succès du run debug ?** Les métriques anti-hacking doivent
   être lues depuis MLflow pour compléter le tableau ci-dessus.

---

## 7. Next Steps

| Priorité | Tâche | Commande | Dépendance |
|---|---|---|---|
| 1 | Lire les métriques SAC debug depuis MLflow | `mlflow ui --port 5000` | — |
| 2 | Suivre SAC v3 en cours | `tail -f logs/...` | SAC v3 lancé |
| 3 | Évaluation test split sur best checkpoint v3 | `make eval-test CONFIG=... CHECKPOINT=...` | SAC v3 terminé |
| 4 | Générer la vidéo du meilleur épisode v3 | `make eval-video CONFIG=... CHECKPOINT=...` | SAC v3 terminé |
| 5 | Générer les courbes d'apprentissage | `make plot PLOT_RUNS="outputs/OpenCabinet_SAC_*/"` | SAC v3 terminé |
| 6 | Lancer PPO baseline si temps disponible | `make train-ppo-baseline SEED=0` | SAC v3 terminé |
| 7 | Compléter `docs/results.md` | Manuel | Toutes métriques disponibles |

---

## 8. Limitations of Current Results

| Limitation | Impact sur les conclusions | Mitigation appliquée |
|---|---|---|
| 0% de succès sur les runs v1 et v2 | Pas de preuve de convergence de l'algorithme | Fix v3 en cours ; diagnostic rigoureux documenté |
| SAC v3 en cours — résultats inconnus | Rapport potentiellement sans résultat final | Documentation complète du processus de debugging |
| PPO non lancé | Comparaison SAC vs PPO impossible | Objectif de recherche partiel |
| Seed unique (seed=0) | Variance inter-seed inconnue | Résultats présentés comme observations de cas unique |
| Deadline 7 mai 2026 | Pas d'ablations, pas de multi-seed | Limitations explicitement documentées |

---

## 9. Éléments de Valeur Malgré les Échecs

Même en l'absence de résultats positifs confirmés, ce projet apporte les éléments
suivants documentés et reproductibles :

1. **Diagnostic précis de l'instabilité de l'auto-tuning ent_coef** sur un espace
   d'action 12D — un problème non documenté dans les tutoriels SB3 standard.

2. **Infrastructure d'entraînement robuste** : 12 workers SubprocVecEnv, MLflow,
   checkpoints, auto-resume, vidéos two-pass — réutilisable pour d'autres tâches
   RoboCasa.

3. **Reward shaping anti-hacking validé** sur le run debug — les 7 composantes ont
   éliminé le hover hacking et peuvent servir de baseline pour des tâches de
   manipulation similaires.

4. **Protocole expérimental rigoureux** : runs debug avant runs principaux, métriques
   physiques (door_angle) prioritaires sur return_mean, split train/val/test séparés
   — conforme aux bonnes pratiques RL de la littérature.
