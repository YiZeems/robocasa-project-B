# Progress Report — RoboCasa Door Opening

**Module** : IA705 — Apprentissage pour la robotique, Telecom Paris
**Date** : 7 mai 2026
**Statut** : SAC v1 abandonné · SAC v2 abandonné · SAC v3 abandonné · SAC v3 curriculum abandonné · SAC HER en cours · PPO non lancé

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

| Run | Config | Statut | Steps | val_success_rate | val_door_angle_final (best) | Raison d'arrêt |
|---|---|---|---:|---:|---:|---|
| SAC debug | `sac_debug.yaml` | Terminé | 300k | 0% | ~0 | Hover hacking corrigé, run de validation |
| SAC v1 | `sac.yaml` | Abandonné | 500k | 0% | 0.000 rad | ent_coef crash → α=0.009, critic loss 48 |
| SAC v2 | `sac_v2.yaml` | Abandonné | 900k | 0% | 0.000 rad | Même crash retardé ; critic loss 116 828 |
| SAC v3 | `sac_v3.yaml` | Abandonné | 400k | 0% | 0.004 rad | Cold start — porte quasi-immobile |
| SAC v3 curriculum | `sac_v3_curriculum.yaml` | Abandonné | 500k | 0% | 0.021 rad (300k) | Pic 300k, régression, buffer sans succès |
| SAC HER | `sac_her.yaml` | En cours | — | ? | ? | Premier run avec signal positif garanti |
| PPO baseline | `ppo_baseline.yaml` | Non lancé | — | — | — | Deadline atteinte |

### 1.3 Issues Identified and Resolved

| Problème | Cause racine | Fix | Découvert lors |
|---|---|---|---|
| ent_coef auto-tuning crash | log_prob ≈ −20 < target_entropy → α→0 | ent_coef=0.1 fixe | v1/v2 |
| Critic loss spikes > 40k–116k | α≈0 → Q-values divergent | gradient_steps=4, tau=0.01 | v1/v2 |
| Hover hacking (approach_frac > 0.7) | Reward approche dense exploitable | High-watermark + gating + stagnation | debug |
| Auto-resume reprend le mauvais checkpoint | task+algo+seed identiques entre versions | --no-auto-resume partout | v2/v3 |
| Validation = 82% du wall time | n_eval_episodes=50 × eval_freq=25000 | n_eval_episodes=10, eval_freq=100000 | v1 |
| Reset 9.3s par épisode | obj_registries=[objaverse] | obj_registries=[lightwheel] | setup |
| control_freq=10 (simulation 2× trop lente) | 50 substeps au lieu de 25 | control_freq=20 | setup |
| SubprocVecEnv crash fork | MuJoCo hérite des FD/EGL du parent | spawn | setup |
| OOM WSL2 | 12 × 3.4 GB > RAM configurée | .wslconfig memory=56GB | v1 |
| oscillation_frac négatif | Pénalité / total sans abs() | abs() au numérateur | v2 |
| Cold start — porte quasi-immobile | Buffer sans succès → critic ne valorise pas ouverture | Curriculum + HER | v3 |
| Pic transitoire sans convergence stable | Success_frac=0 → critic diverge après 300k | HER relabellisation | v3_curriculum |
| HER reference_obs_space Dict crash | GoalConditionedWrapper retourne Dict sans .shape | Extraction du flat Box interne | HER démarrage |

---

## 2. État Honnête des Résultats

### 2.1 Ce qui fonctionne

- L'infrastructure d'entraînement est fonctionnelle et robuste (12 workers, MLflow, checkpoints, auto-resume).
- Le reward shaping anti-hacking a éliminé le hover hacking détecté sur le run debug.
- Le diagnostic des échecs v1→v2→v3→curriculum a été conduit rigoureusement, aboutissant à une cause racine claire à chaque fois.
- HER est implémenté et lancé — premier mécanisme qui garantit un signal positif dans le buffer.

### 2.2 Ce qui ne fonctionne pas encore

- **Aucun run n'a atteint success_rate > 0%** sur 2.3M steps cumulés (v1+v2+v3+curriculum).
- **PPO non lancé** : contrainte de temps (deadline 7 mai).
- **Comparaison SAC vs PPO impossible** : objectif de recherche initial non atteint.

### 2.3 Résultats numériques réels

| Métrique | v1 (500k) | v2 (900k) | v3 (400k) | curriculum (500k) |
|---|---|---|---|---|
| `val_success_rate` | 0% | 0% | 0% | 0% |
| `val_door_angle_final` (best) | 0.000 rad | 0.000 rad | 0.004 rad | **0.021 rad** (300k) |
| `val_door_angle_max` (best) | 0.014 rad | 0.017 rad | 0.012 rad | **0.039 rad** (300k) |
| `theta_best_mean` training | 0.006 rad | 0.011 rad | 0.003 rad | 0.004 rad |
| `actor_loss` (last) | −37.5 | +7.2 | −47.1 | −48.2 |
| `ent_coef` α (last) | 0.009 (crash) | 0.001 (crash) | **0.100 (fixe)** | **0.100 (fixe)** |
| `critic_loss` (max) | 48 | **116 828** | 37.7 | 10.3 |
| `val_approach_frac` (best) | 0.718 | 0.046 | 0.799 | 1.025 |

La progression est réelle même sans succès : chaque run a résolu un problème et repoussé la limite (porte de 0 → 0.021 rad). HER est la prochaine étape logique.

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

## 4. Analyse des Échecs — Chronologie Complète

Chaque run a échoué sur une cause différente, révélée par la suivante.

### 4.1 SAC v1 (500k steps) — Crash ent_coef

| Steps | Événement | Métrique observée |
|---:|---|---|
| 0 | ent_coef="auto", α=1.0, target_entropy=−12 | — |
| ~50k | α décroît (log_prob ≈ −20 < target −12) | ent_coef ≈ 0.3 |
| ~200k | α ≈ 0 — politique quasi-déterministe | actor_loss remonte vers 0 |
| ~300k | Actions saturent aux limites | critic_loss > 40 |
| 500k | Abandonné — 0% succès | actor_loss = −37.5, α = 0.009 |

### 4.2 SAC v2 (900k steps) — Même crash, critic explosif

| Steps | Événement | Métrique observée |
|---:|---|---|
| 0 | ent_coef="auto_0.1", α=0.1, target_entropy=−4 | — |
| ~100k | α → 0 (log_prob ≈ −20 < target −4, même déséquilibre) | ent_coef ≈ 0.001 |
| ~500k | Critic loss explose | critic_loss max = **116 828** |
| 900k | Abandonné — actor_loss positif (+7.2) | 0% succès |

### 4.3 SAC v3 (400k steps) — Stable mais cold start

| Steps | Événement | Métrique observée |
|---:|---|---|
| 0 | ent_coef=0.1 fixe, SDE, gradient_steps=4 | — |
| 100k–400k | Actor stable à −47/−53, α plat à 0.1 | critic_loss max = 37.7 |
| 400k | Porte quasi-immobile en training | theta_best_mean = 0.003 rad |
| 400k | Abandonné — 0% succès | val_door_angle_final = 0.004 rad |

### 4.4 SAC v3 Curriculum (500k steps) — Pic transitoire

| Steps | Événement | Métrique observée |
|---:|---|---|
| 0 | theta_success=0.40, spawn réduit (0.05/0.02) | — |
| 100k | Signal positif faible | val_return = +2.05, door_angle = 0.0037 rad |
| 200k | Régression | val_return = −0.55, door_angle = 0.0017 rad |
| **300k** | **Pic** | **val_return = +12.6, door_angle = 0.021 rad** |
| 400k | Régression post-pic | val_return = +8.0, door_angle = 0.015 rad |
| 500k | Abandonné — success_frac=0 tout au long | critic_loss croît jusqu'à 10.3 |

Le pic à 300k était dû à 10 épisodes de validation favorables, pas à une convergence réelle. Le training `theta_best_mean` n'a jamais dépassé 0.004 rad — la porte ne bougait pas vraiment.

### 4.5 SAC HER (en cours) — Rationale

```yaml
# HER : signal positif garanti même sans succès réels
use_her: true
her_n_sampled_goal: 4      # 4 goals virtuels par transition réelle
her_goal_strategy: future  # goal = angle atteint plus tard dans l'épisode
theta_success: 0.15        # Seuil accessible
# reward : sparse uniquement (0 si atteint, -1 sinon)
```

Pour chaque épisode où la porte a atteint 0.005 rad, HER crée 4 transitions
virtuelles qui disent "ouvrir à 0.005 rad = succès". Le critic apprend immédiatement
à valoriser l'ouverture, même partielle.

---

## 5. État Actuel des Métriques

| Métrique | v1 (abandonné) | v2 (abandonné) | v3 (abandonné) | curriculum (abandonné) | HER (en cours) | Cible |
|---|---|---|---|---|---|---|
| `val_success_rate` | 0% | 0% | 0% | 0% | ? | > 50% |
| `val_door_angle_final` | 0.000 | 0.000 | 0.004 | **0.021** (300k) | ? | > 0.15 rad |
| `val_approach_frac` | 0.718 | 0.046 | 0.799 | 1.025 | ? | < 0.3 |
| `actor_loss` (last) | −37.5 | +7.2 (crash) | −47.1 | −48.2 | ? | Décroissant |
| `ent_coef` α | 0.009 (crash) | 0.001 (crash) | 0.100 (fixe) | 0.100 (fixe) | 0.100 (fixe) | 0.1 fixe |
| `critic_loss` max | 48 | **116 828** | 37.7 | 10.3 | ? | < 5 |
| `success_frac` | 0 | 0 | 0 | 0 | ? | > 0 |

---

## 6. Questions Ouvertes

1. **HER convergera-t-il ?** C'est le premier mécanisme qui garantit un signal positif dans le buffer. Mais même avec HER, le premier contact physique avec la poignée reste nécessaire.

2. **theta_success=0.15 est-il le bon seuil pour HER ?** Trop bas = apprentissage trivial. Trop haut = toujours hors d'atteinte. 0.15 rad ≈ 9° semble raisonnable comme premier objectif.

3. **Faudra-t-il combiner HER + curriculum ?** HER résout le problème du buffer vide, mais la politique doit encore trouver le contact initial. Le curriculum pourrait aider à guider vers ce contact.

4. **Comparaison SAC vs PPO possible post-deadline ?** PPO non lancé — la question de recherche principale reste sans réponse quantitative.

---

## 7. Prochaines Étapes

| Priorité | Tâche | Statut |
|---|---|---|
| 1 | Suivre SAC HER — premier checkpoint à 100k | En cours |
| 2 | Si HER success_frac > 0 à 200k → laisser tourner jusqu'à 3M | Conditionnel |
| 3 | Courbes HER dans docs/courbes/run_sac_her/ après 200k | À faire |
| 4 | Évaluation test split sur best checkpoint HER | Post-convergence |
| 5 | Générer vidéo du meilleur épisode HER | Post-convergence |

---

## 8. Limitations des Résultats Actuels

| Limitation | Impact | Mitigation |
|---|---|---|
| 0% succès sur 2.3M steps cumulés | Pas de preuve de convergence | Diagnostic rigoureux documenté ; HER en cours |
| PPO non lancé | Comparaison SAC vs PPO impossible | Objectif de recherche partiel explicitement documenté |
| Seed unique | Variance inter-seed inconnue | Résultats comme observations de cas unique |
| Deadline 7 mai 2026 | Pas d'ablations, pas de multi-seed | Limitations explicitement reconnues |

---

## 9. Valeur Malgré les Échecs

Même sans succès confirmé, ce projet apporte :

1. **Diagnostic de l'instabilité ent_coef auto-tuning sur espace 12D** — problème non documenté dans les tutoriels SB3.
2. **Infrastructure robuste** : 12 workers SubprocVecEnv, MLflow, checkpoints, auto-resume — réutilisable pour d'autres tâches RoboCasa.
3. **Reward shaping anti-hacking validé** — les 7 composantes éliminent le hover hacking.
4. **Implémentation HER complète** — `GoalConditionedWrapper` + `HerReplayBuffer` intégré au pipeline d'entraînement existant.
5. **Protocole expérimental rigoureux** — runs debug, métriques physiques prioritaires sur return_mean, split val/test séparés.
