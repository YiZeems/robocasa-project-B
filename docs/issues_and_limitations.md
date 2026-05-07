# Issues and Limitations

> This document covers both **technical issues encountered and resolved** during development, and **inherent limitations** of the current project design that would require future work to address.

---

## 1. Technical Issues Encountered and Resolved

### Summary Table

| # | Catégorie | Problème | Symptôme | Cause | Solution / mitigation | Statut |
|---|---|---|---|---|---|---|
| 1.1 | Exploration / SAC | `ent_coef` auto-tuning crash (v1 et v2) | `actor_loss` remonte vers 0, actions saturent aux limites, `return_mean` descend de −15 à −35 | `log_prob ≈ −20` toujours < `target_entropy` → SAC pousse `α` vers 0 systématiquement | `ent_coef=0.1` fixe (v3) — suppression de l'auto-tuning | Résolu (v3) |
| 1.2 | Stabilité critic | Critic loss spikes (> 40 000 en v2, > 37 en v3) | Divergence `critic_loss`, `actor_loss` instable | `α ≈ 0` → absence de régularisation entropique → Q-values divergent | `gradient_steps=4`, `tau=0.01` | Résolu (v2/v3) |
| 1.3 | Reward hacking | Hover hacking (approche sans ouverture) | `approach_frac > 0.7`, `val_success_rate = 0%`, return ≈ 200–400 | Reward d'approche dense exploitable sans condition de progression | High-watermark progress + gating approche + pénalité stagnation + dominance succès | Résolu |
| 1.4 | Auto-resume | Auto-resume reprend le mauvais checkpoint (v1/v2/v3) | Run v3 reprend le checkpoint cassé de v1 ou v2 | `task + algo + seed` identiques entre versions → `auto_resume` sélectionne le dernier run compatible | `--no-auto-resume` dans tous les targets Makefile | Résolu |
| 1.5 | Auto-resume | Boucle infinie de reprise après crash | Crash loop sur un run qui a écrit `final_model.zip` mais pas `train_summary.json` | `_run_is_complete()` vérifie uniquement `train_summary.json` | Fallback sur `final_model.json` + comparaison `num_timesteps >= target` | Résolu |
| 1.6 | Performance | Validation = 82 % du wall time (v1) | Temps d'entraînement ~8× plus long qu'attendu | `n_eval_episodes=50` × `eval_freq=25000` = 23 min par validation | `n_eval_episodes=5→10`, `eval_freq=100000` | Résolu (v2/v3) |
| 1.7 | Performance | Reset 9.3 s par épisode | Workers lents, faible throughput | `obj_registries=[objaverse]` charge des assets lourds | `obj_registries=[lightwheel]` → reset ~4.8 s | Résolu |
| 1.8 | Performance | `control_freq` mal configuré | Simulation 2× plus lente que nécessaire | `control_freq=10` (50 substeps MuJoCo) au lieu de 20 (25 substeps) | `control_freq=20` | Résolu |
| 1.17 | Apprentissage | Cold start — aucun contact avec la poignée (v3) | `theta_best_mean` ≈ 0.002–0.003 rad à 400k steps, porte quasi immobile | Pas de premier signal positif → replay buffer vide de succès → critic ne valorise pas l'ouverture | Curriculum learning : `theta_success=0.40`, spawn déviation réduite (v3_curriculum) | Partiellement résolu |
| 1.18 | Apprentissage | Pic transitoire sans apprentissage stable (v3_curriculum) | `val_door_angle_final` atteint 0.021 rad à 300k puis redescend à 0.015–0.017 à 500k | Buffer toujours sans succès confirmés (`success_frac=0`) → critic diverge (loss 0→6+) → régression | HER : relabellisation rétroactive des goals → succès virtuels créés à chaque épisode | En cours (HER) |
| 1.19 | Apprentissage | Replay buffer sans signal positif | `success_frac=0` sur toutes les runs → critic ne peut pas apprendre à valoriser l'ouverture | Récompense sparse jamais déclenchée même avec `theta_success=0.40` sur 500k steps | HER avec `n_sampled_goal=4`, `strategy=future`, `theta_success=0.15` | En cours (HER) |
| 1.20 | Infrastructure | HER : `reference_obs_space` Dict transmis à `RawRoboCasaAdapter` | `TypeError: int() argument must be a string…NoneType` au démarrage | `GoalConditionedWrapper.observation_space` est Dict (sans `.shape`) ; les workers recevaient le Dict au lieu du Box interne | Extraction du flat Box space via `getattr(_ref_env, '_base', None)` avant passage aux workers | Résolu |
| 1.9 | Infrastructure | `SubprocVecEnv` crash `ConnectionResetError` au démarrage | Workers meurent immédiatement | `fork` : MuJoCo hérite des FD/EGL du process parent → contexte corrompu | Utiliser `spawn` (défaut SB3) | Résolu |
| 1.10 | Infrastructure | OOM WSL2 (même symptôme `ConnectionResetError`) | Workers crashent aléatoirement pendant l'entraînement | 12 workers × 3.4 GB = ~41 GB > RAM WSL2 par défaut (31 GB) | `.wslconfig memory=56GB` | Résolu |
| 1.11 | Infrastructure | Vidéo impossible pendant l'entraînement | Crash EGL/OpenGL dans les workers | `SubprocVecEnv` : contextes OpenGL non partagés entre processus | Two-pass `eval_video.py` : scoring sans rendu + reproduction avec seed identique | Résolu |
| 1.12 | Infrastructure | `VecMonitor.reset()` crash | `TypeError: unexpected keyword argument 'seed'` | API Gymnasium ≥ 0.26 vs implémentation `VecMonitor` | Détection `isinstance(env, VecEnv)` ; `env.seed(n); env.reset()` pour VecEnvs | Résolu |
| 1.13 | Infrastructure | `mlruns/` committé dans Git | Repo alourdi par des artefacts binaires | Répertoire oublié dans `.gitignore` avant premier commit | `git rm -r --cached mlruns/` + hook pre-commit | Résolu |
| 1.14 | Infrastructure | Observation shape instable entre resets | Buffer SB3 pré-alloué avec mauvaise shape → crash | `GymWrapper` RoboSuite varie selon la scène randomisée | `RawRoboCasaAdapter` : `flatten()` avec espace de référence fixe à la construction | Résolu |
| 1.15 | Monitoring | `oscillation_frac` négatif dans MLflow | Fraction apparaît négative dans les courbes | Pénalité (reward négatif) divisée par `abs(total_mean)` sans `abs()` au numérateur | `abs()` appliqué au numérateur | Résolu |
| 1.16 | Monitoring | Métriques insuffisantes si on lit seulement la reward | Agent semble progresser alors que `success_rate = 0%` | Return négatif même quand l'agent améliore son comportement de base | Ajout de `door_angle_final`, `success_rate`, `theta_best` comme métriques primaires | Résolu |

---

### 1.1 Ent_coef Auto-tuning Crash (SAC v1 et v2)

**Description :** L'auto-tuning du coefficient d'entropie `α` de SAC échoue
systématiquement sur cette tâche, rendant la politique déterministe avant qu'elle
ait le temps d'apprendre.

- **v1** (`ent_coef="auto"`) : `α` démarre à 1.0, `target_entropy = −action_dim = −12`.
  La `log_prob` de la politique initiale (~−20) est systématiquement inférieure
  au target → SAC pousse `α` vers 0 dès les 200k premiers steps.
- **v2** (`ent_coef="auto_0.1"`, `target_entropy=−4`) : `α` démarre plus bas, mais
  la `log_prob` reste inférieure au target → même crash vers 0 dès 100k steps.

**Symptômes observés :** `actor_loss` remonte de valeurs négatives vers 0 puis
positif ; actions saturent aux bornes de l'action space ; `return_mean` régresse
de −15 vers −35 ; `val_success_rate = 0%` à 700k steps (v1) et 100k steps (v2).

**Cause profonde :** Le `target_entropy` est trop agressif pour une politique en
cours d'initialisation. La log-prob d'une politique aléatoire sur un espace 12D
est structurellement inférieure à −4 ou −12, ce qui fait toujours décroître `α`.

**Fix v3 :** `ent_coef=0.1` fixe + `use_sde=True`. Suppression totale de
l'auto-tuning. `α = 0.1` permanent garantit un signal d'exploration tout au long
de l'entraînement.

**Lien cours :** Exploration vs exploitation — le coefficient d'entropie contrôle
directement la diversité de la politique (SAC = maximum entropy RL).

**Statut :** Résolu (v3 en cours).

---

### 1.2 Critic Loss Spikes

**Description :** À partir de 500k steps, la `critic_loss` explose au-delà de
40 000 par intervalles, déstabilisant l'`actor_loss`.

**Cause :** En l'absence d'entropie (`α ≈ 0`), les Q-values ne sont plus
régularisées. Le replay buffer contient des transitions anciennes surestimées →
bootstrapping instable.

**Fix :** `gradient_steps=4` (moins d'overfit sur le critic à chaque step) et
`tau=0.01` (target network suit le critic plus rapidement → overestimation réduite).

**Lien cours :** Actor-critic — la stabilité du critic est le prérequis de la
qualité des gradients de l'acteur.

---

### 1.3 Hover Hacking (Reward Gaming)

**Description :** L'agent apprend à rester immobile à ~5 cm de la poignée de porte
et accumule la reward d'approche sans jamais ouvrir la porte. Observé sur le run
debug. Métriques : `return_mean ≈ 200–400`, `val_success_rate = 0%`,
`approach_frac > 0.7`.

**Cause :** La reward d'approche (`r_approach ∝ 1 − d/d_max`) est dense et
exploitable. Le bonus de succès (sparse) est trop rare pour être découvert pendant
l'exploration initiale.

**Fix :** Refonte multi-couches de la reward (voir `docs/reward_shaping.md`) :
high-watermark progress, gating de l'approche, pénalité stagnation, dominance
du bonus succès (100× la reward d'approche maximale).

**Lien cours :** Reward shaping — le façonnage de récompense peut introduire un
biais (Ng et al., 1999). Il faut toujours monitorer `val_success_rate` comme
métrique primaire, jamais le `return_mean` seul.

---

### 1.4–1.16 Voir le tableau récapitulatif ci-dessus.

---

### 1.17 Cold Start — Aucun Contact avec la Poignée (v3)

**Description :** Même avec `ent_coef=0.1` fixe et SDE, v3 n'a jamais établi de
contact utile avec la poignée après 400k steps. La métrique `theta_best_mean` en
training oscillait entre 0.0017 et 0.0027 rad — soit environ 0.15 degrés d'ouverture,
probablement des vibrations mécaniques et non un vrai tirage.

**Métriques observées (v3 @ 400k) :**
- `train/actor_loss` : stable à −47 → −53 (bon signe, pas de crash)
- `train/ent_coef` : plat à 0.1 (fix confirmé)
- `train/critic_loss` : max 37.7 (contenu mais croissant)
- `val_door_angle_final_mean` : 0.0037 rad (meilleur checkpoint, 400k)
- `val_success_rate` : 0% à tout checkpoint

**Cause :** La tâche requiert un contact précis avec une poignée de ~5 cm. Pendant
l'exploration SDE, le bras bouge de façon cohérente mais aléatoire. La probabilité
d'un contact productif sur 500 steps est structurellement faible. Sans ce premier
signal de progression, le replay buffer ne contient que des transitions à reward
négative ou nulle → le critic ne peut pas apprendre que l'ouverture de la porte est
désirable → l'actor ne cherche pas à ouvrir.

**Fix appliqué (v3_curriculum) :** Réduire le seuil de succès à `theta_success=0.40`
(40% d'ouverture vs 90%) et réduire la variance de spawn du robot (`pos_x: 0.05`,
`pos_y: 0.02`) pour que l'agent voie des scènes similaires et que le critic
converge plus vite.

**Lien cours :** Exploration et crédit à long terme (credit assignment). Sans
signal de récompense positive, les méthodes off-policy comme SAC ne peuvent pas
apprendre même avec un replay buffer plein.

---

### 1.18 Pic Transitoire sans Convergence (v3_curriculum)

**Description :** v3_curriculum a montré un signal positif fort à 300k steps
(`val_return_mean` +12.6, `val_door_angle_final_mean` 0.021 rad) puis a régressé
aux checkpoints 400k et 500k (0.015 et 0.017 rad).

**Métriques observées (v3_curriculum) :**

| Step | val_return | val_door_angle_final | val_approach_frac | critic_loss (max) |
|---:|---:|---:|---:|---:|
| 100k | +2.05 | 0.0037 | 0.60 | ~2 |
| 200k | −0.55 | 0.0017 | 0.47 | ~4 |
| 300k | +12.60 | 0.0213 | 1.03 | ~5 |
| 400k | +8.0 | 0.0153 | — | ~6 |
| 500k | +9.4 | 0.0169 | — | ~6+ |

Le pic à 300k correspond à 10 épisodes de validation déterministes qui ont "eu de
la chance" sur une politique légèrement meilleure. Ce n'est pas de la convergence :
`theta_best_mean` en training restait à 0.0021–0.0038 rad (quelques dixièmes de
degré), et `success_frac = 0` tout au long (aucun épisode n'a jamais atteint
`theta_success=0.40` même avec exploration).

**Cause profonde :** Le replay buffer contient 500k transitions sans aucun succès.
La critic_loss croît de façon monotone (0 → 6+) car le critic tente d'apprendre
des Q-values qui n'ont jamais été positives. L'actor est guidé par un critic de
plus en plus instable → régression après 300k.

**Fix appliqué (HER) :** Hindsight Experience Replay — après chaque épisode, 4
transitions virtuelles sont créées pour chaque transition réelle avec un objectif
relabellisé = un angle atteint plus tard dans le même épisode. Si la porte a
atteint 0.02 rad, des succès virtuels sont créés pour `desired_goal=0.02 rad` →
le critic apprend immédiatement à valoriser l'ouverture de porte.

**Lien cours :** HER (Andrychowicz et al., 2017) — technique standard pour les
tâches de manipulation avec reward sparse.

---

### 1.19 Replay Buffer Sans Signal Positif

**Description :** Sur les 4 runs (v1 à v3_curriculum), la métrique
`reward_hack/success_frac` est restée à 0 sur toute la durée. Aucun épisode
d'entraînement n'a jamais déclenché le bonus de succès, même avec `theta_success`
abaissé à 0.40.

**Impact :** En off-policy RL, le critic estime `Q(s,a) = r + γ·V(s')`. Sans
transitions avec `r > 0` liées à l'ouverture de la porte, `Q(s_handle, a_pull)`
reste négatif ou nul → l'actor n'est jamais guidé vers la saisie. Les 500k–900k
transitions dans le buffer décrivent un monde où ouvrir la porte ne rapporte rien.

**Fix HER :** `GoalConditionedWrapper` retourne une observation augmentée
`{observation, achieved_goal=[theta], desired_goal=[theta_success]}`.
`HerReplayBuffer` crée 4 transitions virtuelles avec `desired_goal` = angles futurs
atteints dans l'épisode. `compute_reward` retourne `0` si `achieved >= desired`,
`-1` sinon. Résultat : pour chaque épisode où la porte a atteint 0.005 rad,
le buffer reçoit des transitions qui disent "ouvrir à 0.005 rad = succès".

---

### 1.20 HER : reference_obs_space Dict Transmis aux Workers

**Description :** Au démarrage de `train-sac-her`, tous les workers SpawnProcess
crashaient avec `TypeError: int() argument must be a string…NoneType`.

**Cause :** Dans `main()`, le code faisait :
```python
_ref_env = make_env_from_config(ctx.env_cfg, seed=ctx.seed)
reference_obs_space = _ref_env.observation_space  # Dict space avec use_her=True
```
Ce `reference_obs_space` (Dict, sans `.shape`) était transmis à `RawRoboCasaAdapter`
qui appelle `int(np.prod(reference_obs_space.shape))` → `shape=None` → crash.

**Fix :**
```python
_inner = getattr(_ref_env, "_base", None)
reference_obs_space = (
    _inner.observation_space if _inner is not None else _ref_env.observation_space
)
```
Les workers reçoivent maintenant le flat Box space interne (248D), puis
`GoalConditionedWrapper` est appliqué par-dessus dans `make_env_from_config`.

---

## 2. Limitations Related to Method Choices

### Summary Table

| # | Catégorie | Limitation | Impact | Mitigation possible |
|---|---|---|---|---|
| 2.1 | Reward | Reward shaping biaise l'apprentissage | Agent optimise la reward shapée, pas nécessairement la tâche | Ablations par composante ; reporter `val_success_rate` comme métrique primaire |
| 2.2 | Algorithme | SAC sensible aux hyperparamètres | Résultats non généralisables sans tuning systématique | Grid search / Optuna sur `lr`, `ent_coef`, `gradient_steps` |
| 2.3 | Simulation | Simulation MuJoCo ≠ monde réel | Politique inutilisable sur robot réel sans adaptation | Domain randomization, system identification |
| 2.4 | Tâche | Contact précis = exploration difficile | Des milliers d'épisodes gaspillés avant le premier contact utile | Imitation learning, curriculum learning |
| 2.5 | Infrastructure | 12 workers compliquent le debugging | Crash silencieux, logs entrelacés, 12× la RAM | Réduire à 4 workers pour debug ; seed tracking par worker |
| 2.6 | Expérimental | Seed unique → variance inconnue | "SAC > PPO" peut être un artefact de seed | 3–5 seeds par algorithme avec intervalles de confiance |
| 2.7 | Expérimental | Pas d'ablations sur les composantes reward | Impact individuel de chaque pénalité inconnu | Ablations : supprimer une composante à la fois, mesurer `approach_frac` et `success_rate` |
| 2.8 | Tâche | Navigation exclue (base fixe) | Tâche simplifiée vs déploiement réel | Intégrer le contrôle de la base mobile |
| 2.9 | Compute | Contrainte de temps (deadline 7 mai 2026) | Un seul seed, pas d'ablations, pas de tuning complet | Résultats présentés comme observations de cas unique |
| 2.10 | Runs v1/v2/v3/curriculum | 0 % de succès sur 500k (v1), 900k (v2), 400k (v3), 500k (curriculum) | Toutes les runs abandonnées avant convergence ; HER en cours | HER (`GoalConditionedWrapper` + `HerReplayBuffer`) |

---

### 2.1 Reward Shaping May Bias Learning

Le reward shaping introduit un **biais de reward shapée** : l'agent optimise la
reward shapée plutôt que l'objectif réel. Même avec les gardes anti-hacking, il
est théoriquement possible qu'un agent trouve une politique qui maximise la reward
shapée avec un `success_rate` inférieur à celui obtenu avec une reward sparse pure.

**Mitigation :** `val_success_rate` est la métrique primaire du rapport. Le
`return_mean` n'est jamais interprété seul. Les métriques `approach_frac`,
`stagnation_steps_mean`, `sign_changes_mean` permettent de détecter une
exploitation de la reward shapée.

**Lien cours :** Reward shaping (Ng et al., 1999) — invariance par potential
shaping vs biais potentiel des composantes non-potentielles.

---

### 2.2 SAC Requires Significant Tuning

SAC comporte plus d'hyperparamètres que PPO : `learning_rate`, `buffer_size`,
`batch_size`, `tau`, `gradient_steps`, `ent_coef`, `learning_starts`,
`target_update_interval`. La configuration actuelle est basée sur les pratiques
de la littérature et les diagnostics des runs v1/v2/v3, mais pas sur une recherche
systématique. La sensibilité aux hyperparamètres est inconnue.

---

### 2.3 MuJoCo Simulation Does Not Capture the Real World

MuJoCo ne modélise pas : variabilité de friction, bruit de capteurs, jeu mécanique
des joints, ni l'apparence visuelle. Une politique entraînée en simulation
échouerait sur un vrai PandaOmron sans domain randomization et adaptation.

**Impact :** Les résultats sont valides uniquement pour l'environnement de
simulation. Aucune conclusion sur les performances en conditions réelles ne peut
être tirée.

---

### 2.4 Contact Learning Is Inherently Difficult

La saisie de la poignée (~5 cm de diamètre) requiert un contact précis avec
application de force dans la bonne direction. En exploration aléatoire, les
contacts utiles sont très rares. Ce problème est intrinsèquement sample-inefficient.

**Lien cours :** Credit assignment — les récompenses tardives (ouverture de porte)
sont difficiles à attribuer aux actions initiales (approche de la poignée) sur
des horizons de 500 steps.

---

### 2.5 12 Workers Complicate Debugging

Un crash dans un worker peut ralentir silencieusement l'entraînement sans erreur
dans le processus principal. Les logs de 12 workers sont entrelacés. La RAM est
amplifiée 12×.

---

### 2.6 Single Seed — Unknown Variance

Avec un seul seed par algorithme, il est impossible de distinguer entre "SAC est
meilleur que PPO sur cette tâche" et "ce seed SAC a eu de la chance". L'intervalle
de confiance Wilson sur N épisodes au sein d'un même run ne quantifie que la
variance intra-run, pas la variance inter-seed.

**Mitigation dans le rapport :** présenter les résultats comme observations de
cas unique ; comparer les tendances qualitatives ; ne pas formuler de conclusions
statistiques fortes.

---

### 2.7 No Ablations on Reward Components

Le reward shaping a 7 composantes et ~14 hyperparamètres. Sans ablations, il est
impossible de savoir quelles composantes contribuent à prévenir le reward hacking
et lesquelles sont redondantes ou nuisibles.

---

### 2.8 Navigation Excluded

La base mobile est fixée. Le robot est toujours positionné face au placard. En
déploiement réel, le robot devrait naviguer depuis une position arbitraire, ce qui
interagit avec la tâche de manipulation.

---

### 2.9 Compute Constraints

| Contrainte | Impact |
|---|---|
| Un seul GPU WSL2 (RTX 4070 ou équivalent) | Un seul seed par algorithme |
| Deadline : 7 mai 2026 | Pas d'ablations, pas de multi-seed, curriculum exclu |
| RAM WSL2 (même après fix à 56 GB) | Impossible de lancer plusieurs runs longs en parallèle |

---

### 2.10 Toutes les Runs Abandonnées — 0% Success Rate

| Run | Steps | `val_success_rate` | `val_door_angle_final` (best) | `theta_best_mean` (training) | `critic_loss` max | Cause d'abandon |
|---|---:|---:|---:|---:|---:|---|
| SAC v1 | 500k | 0% | 0.000 rad | 0.006 rad | 48.0 | `ent_coef` auto crash → α→0.009 à 500k |
| SAC v2 | 900k | 0% | 0.000 rad | 0.011 rad | 116 828 | Même crash retardé ; critic loss explose |
| SAC v3 | 400k | 0% | 0.004 rad | 0.003 rad | 37.7 | Cold start — porte quasi-immobile |
| SAC v3 curriculum | 500k | 0% | 0.021 rad (300k) | 0.004 rad | 10.3 | Pic transitoire, régression, buffer sans succès |
| SAC HER | En cours | ? | ? | ? | ? | En cours |

Ces échecs successifs constituent un résultat expérimental en soi : ils démontrent
que SAC standard ne converge pas sur cette tâche de manipulation précise sans un
mécanisme de création de signal positif (HER). Chaque run a néanmoins résolu un
problème spécifique et affiné le diagnostic :

- **v1** : identifie le crash d'auto-tuning de `ent_coef`
- **v2** : confirme que le problème est structurel (pas juste l'initialisation)
- **v3** : prouve que stabiliser `α` ne suffit pas — cold start est le vrai obstacle
- **curriculum** : prouve que baisser le seuil de succès ne suffit pas — le buffer reste vide de succès
- **HER** : adresse la cause racine — crée du signal positif sans succès réels

---

## 3. Known Warnings (Non-Blocking)

Les warnings suivants apparaissent à l'initialisation de l'environnement et sont
**attendus et sans impact fonctionnel** :

```
WARNING: mimicgen environments not imported since mimicgen is not installed!
WARNING: mink environments not imported since mink is not installed!
UserWarning: WARN: Gym has been unmaintained since 0.26...
```

Ces warnings proviennent de dépendances optionnelles de RoboCasa et n'affectent
pas la tâche `OpenCabinet` ni aucune fonctionnalité utilisée dans ce projet.
