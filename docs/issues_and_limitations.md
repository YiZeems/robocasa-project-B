# Issues and Limitations

> This document covers both **technical issues encountered and resolved** during development, and **inherent limitations** of the current project design that would require future work to address.

---

## 1. Technical Issues Encountered and Resolved

### Summary Table

| # | Catégorie | Problème | Symptôme | Cause | Solution / mitigation | Statut |
|---|---|---|---|---|---|---|
| 1.1 | Exploration / SAC | `ent_coef` auto-tuning crash (v1 et v2) | `actor_loss` remonte vers 0, actions saturent aux limites, `return_mean` descend de −15 à −35 | `log_prob ≈ −20` toujours < `target_entropy` → SAC pousse `α` vers 0 systématiquement | `ent_coef=0.1` fixe (v3) — suppression de l'auto-tuning | Résolu (v3) |
| 1.2 | Stabilité critic | Critic loss spikes (> 40 000) à partir de 500k steps | Divergence `critic_loss`, `actor_loss` instable | `α ≈ 0` → absence de régularisation entropique → Q-values divergent | `gradient_steps=4`, `tau=0.01` | Résolu (v2/v3) |
| 1.3 | Reward hacking | Hover hacking (approche sans ouverture) | `approach_frac > 0.7`, `val_success_rate = 0%`, return ≈ 200–400 | Reward d'approche dense exploitable sans condition de progression | High-watermark progress + gating approche + pénalité stagnation + dominance succès | Résolu |
| 1.4 | Auto-resume | Auto-resume reprend le mauvais checkpoint (v1/v2/v3) | Run v3 reprend le checkpoint cassé de v1 ou v2 | `task + algo + seed` identiques entre versions → `auto_resume` sélectionne le dernier run compatible | `--no-auto-resume` dans tous les targets Makefile | Résolu |
| 1.5 | Auto-resume | Boucle infinie de reprise après crash | Crash loop sur un run qui a écrit `final_model.zip` mais pas `train_summary.json` | `_run_is_complete()` vérifie uniquement `train_summary.json` | Fallback sur `final_model.json` + comparaison `num_timesteps >= target` | Résolu |
| 1.6 | Performance | Validation = 82 % du wall time (v1) | Temps d'entraînement ~8× plus long qu'attendu | `n_eval_episodes=50` × `eval_freq=25000` = 23 min par validation | `n_eval_episodes=5→10`, `eval_freq=100000` | Résolu (v2/v3) |
| 1.7 | Performance | Reset 9.3 s par épisode | Workers lents, faible throughput | `obj_registries=[objaverse]` charge des assets lourds | `obj_registries=[lightwheel]` → reset ~4.8 s | Résolu |
| 1.8 | Performance | `control_freq` mal configuré | Simulation 2× plus lente que nécessaire | `control_freq=10` (50 substeps MuJoCo) au lieu de 20 (25 substeps) | `control_freq=20` | Résolu |
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
| 2.10 | Runs v1/v2 | 0 % de succès sur 700k steps (v1) et 100k steps (v2) | Runs abandonnés ; seul v3 est potentiellement valide | Fix v3 (`ent_coef=0.1` fixe + SDE) |

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

### 2.10 Runs v1 et v2 Abandonnés — 0% Success Rate

| Run | Steps réalisés | `val_success_rate` | `theta_best_mean` | Cause d'abandon |
|---|---:|---:|---:|---|
| SAC v1 | 700k | 0% | 0.001–0.012 | `ent_coef` crash → politique déterministe → actions saturées |
| SAC v2 | 100k | 0% | À compléter | Même cause ; détectée plus tôt grâce aux métriques renforcées |
| SAC v3 | En cours | À compléter | À compléter | Fix appliqué ; résultat inconnu |

Ces deux échecs constituent un résultat expérimental en soi : ils démontrent que
l'auto-tuning du coefficient d'entropie SAC est instable sur un espace d'action
12D lorsque la `log_prob` de la politique initiale est structurellement inférieure
au `target_entropy`.

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
