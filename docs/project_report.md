# Rapport de projet — Apprentissage par renforcement pour la manipulation robotique

**Module** : IA705 — Apprentissage pour la robotique  
**Formation** : Mastère Spécialisé Intelligence Artificielle Multimodale, Telecom Paris  
**Date de rendu** : 7 mai 2026  
**Repo** : https://github.com/YiZeems/robocasa-project-B

---

## Table des matières

1. [Introduction](#1-introduction)
2. [Contexte : RoboCasa](#2-contexte--robocasa)
3. [Robot : PandaOmron](#3-robot--pandaomron)
4. [Tâche : ouvrir une porte de placard](#4-tâche--ouvrir-une-porte-de-placard)
5. [Formulation RL](#5-formulation-rl)
6. [Méthodes](#6-méthodes)
7. [Reward shaping](#7-reward-shaping)
8. [Configuration expérimentale](#8-configuration-expérimentale)
9. [Métriques suivies](#9-métriques-suivies)
10. [Protocole expérimental](#10-protocole-expérimental)
11. [Résultats](#11-résultats)
12. [Discussion](#12-discussion)
13. [Difficultés rencontrées et solutions](#13-difficultés-rencontrées-et-solutions)
14. [Limites et perspectives](#14-limites-et-perspectives)
15. [Conclusion](#15-conclusion)
16. [Références](#16-références)
17. [Justification of Methods and Tools](#17-justification-of-methods-and-tools)
18. [Relation to Course Concepts](#18-relation-to-course-concepts)

---

## 1. Introduction

L'apprentissage par renforcement (RL) appliqué à la manipulation robotique est un défi ouvert : l'espace d'état est continu et de haute dimension, les récompenses sont rares, et la simulation ne se transfère pas toujours fidèlement au monde réel. Ce projet explore une instance concrète de ce problème : apprendre à un robot à **ouvrir une porte de placard de cuisine** dans le simulateur RoboCasa, sans aucune démonstration préalable, à partir de la seule récompense.

L'enjeu scientifique central est double :

1. **Comparaison d'algorithmes** : SAC (off-policy, maximisation d'entropie) vs PPO (on-policy, gradient de politique proximal) sur une tâche de manipulation continue.
2. **Robustesse de la reward** : comment concevoir un signal de récompense qui guide l'apprentissage sans être exploitable par l'agent d'une manière non souhaitée (*reward hacking*).

---

## 2. Contexte : RoboCasa

**RoboCasa** (Nasiriany et al., 2024) est un simulateur de cuisine procédural construit sur **RoboSuite** (Zhu et al., 2020) et **MuJoCo** (Todorov et al., 2012). Il propose :

- Plus de 100 scènes de cuisine générées procéduralement avec variation d'objets, de textures, et de dispositions.
- 22 tâches atomiques de manipulation (ouvrir une porte, ouvrir un tiroir, saisir un objet, etc.).
- Des tâches composites qui enchaînent plusieurs tâches atomiques.
- Des assets photoréalistes issus de la bibliothèque Objaverse.

**Avantages pour ce projet :**
- Variabilité procédurale → généralisation testable
- Physique MuJoCo fiable → contacts réalistes
- Intégration Gymnasium → compatible avec Stable-Baselines3

**Limites :**
- Pas de transfert sim-to-real documenté pour PandaOmron
- Assets lourds (~4 Go) nécessitent un téléchargement séparé
- Certains wrappers Gym présentent des instabilités d'API entre resets (voir section 13)

---

## 3. Robot : PandaOmron

Le **PandaOmron** est un robot composé de deux parties :

- **Franka Panda** : bras manipulateur 7 DoF avec une pince 2 doigts. Précision submillimétrique, limites d'effort strictes pour la sécurité.
- **Omron LD-250** : base mobile omnidirectionnelle.

**Choix de conception :** dans ce projet, **la navigation est explicitement exclue**. La base reste fixe, positionnée face au placard au moment du reset. Seul le bras est contrôlé. Ce choix est justifié par :

- La deadline académique : réduire l'espace d'action simplifie l'apprentissage.
- La tâche atomique choisie ne nécessite pas de déplacement de base.
- La combinaison navigation + manipulation est un problème de recherche ouvert à part entière.

**Espace d'action effectif :** commande différentielle du bras Panda (7 degrés de liberté), vecteur continu de dimension ~7–8 selon la configuration RoboSuite.

---

## 4. Tâche : ouvrir une porte de placard

### Définition

La tâche `OpenCabinet` (alias `OpenSingleDoor`) consiste à faire pivoter la porte d'un placard de cuisine jusqu'à une ouverture suffisante.

| Propriété | Valeur |
|---|---|
| Nom de tâche RoboCasa | `OpenCabinet` |
| Critère de succès | Angle normalisé θ ≥ 0.90 (90 % de l'ouverture maximale) |
| Horizon d'épisode | 500 steps |
| Fréquence de contrôle | 10 Hz (50 secondes simulées max.) |

### Observation

L'agent reçoit un vecteur d'état bas niveau (~50 scalaires) comprenant :
- Positions et vitesses des joints du bras
- Position et orientation de l'effecteur final (EEF)
- Distance EEF → poignée de porte
- Angle d'ouverture de la porte (θ)
- Indicateur de contact avec la poignée

**Choix délibéré :** les observations RGB (caméra) ont été écartées pour réduire la complexité du problème et tenir la deadline. Avec les observations d'état, l'agent n'a pas à apprendre une représentation visuelle, ce qui accélère l'apprentissage.

### Succès et échec

- **Succès** : θ ≥ 0.90 atteint à un moment quelconque de l'épisode.
- **Échec** : horizon de 500 steps épuisé sans atteindre le critère.
- La fonction de succès est détectée via `infer_success()` — une hiérarchie de sondes qui cherche `info["success"]`, `_check_success()` sur l'environnement brut, puis remonte les wrappers.

---

## 5. Formulation RL

### MDP (Processus de Décision Markovien)

| Élément | Définition |
|---|---|
| **États** S | Vecteur d'observation bas niveau ∈ ℝ^n |
| **Actions** A | Commande différentielle du bras ∈ ℝ^7 (espace continu borné) |
| **Reward** R | Signal façonné (voir section 7) |
| **Horizon** T | 500 steps par épisode |
| **Facteur d'actualisation** γ | 0.99 |
| **Succès** | θ_normalisé ≥ 0.90 |

### Politique apprise

L'agent apprend une politique stochastique π_θ(a|s) modélisée par un réseau MLP [256, 256] — même architecture pour SAC et PPO. La politique déterministe est utilisée à l'évaluation.

---

## 6. Méthodes

### 6.1 SAC — Soft Actor-Critic (méthode principale)

SAC (Haarnoja et al., 2018) est un algorithme off-policy qui maximise simultanément la récompense cumulée et l'entropie de la politique :

```
π* = argmax_π Σ_t E[R(s_t, a_t) + α · H(π(·|s_t))]
```

**Avantages justifiant ce choix :**
- **Off-policy** : réutilise les expériences passées via un replay buffer, ce qui est critique avec seulement 50 secondes simulées par épisode.
- **Maximisation d'entropie** : l'exploration est intrinsèque, sans avoir à définir un calendrier ε-greedy.
- **Sample efficiency** : 3M steps suffisent là où PPO en nécessite souvent 10M+.
- **Compatible manipulation continue** : l'espace d'action continu est géré nativement.

**Implémentation :** Stable-Baselines3 v2.3.2, 12 workers parallèles (SubprocVecEnv), replay buffer de 1M transitions.

### 6.2 PPO — Proximal Policy Optimization (baseline)

PPO (Schulman et al., 2017) est un algorithme on-policy qui contraint les mises à jour de politique via un ratio clipé :

```
L_CLIP(θ) = E[min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t)]
```

**Rôle ici :** baseline comparative. PPO est plus simple à tuner et plus stable, mais moins sample-efficient sur les tâches de manipulation continue.

**Raison de la comparaison :** montrer que SAC converge avec moins de steps et vers un taux de succès plus élevé, justifiant son choix comme méthode principale.

### 6.3 Imitation learning

L'imitation learning (Behavior Cloning, Diffusion Policy) n'a **pas** été utilisé dans ce projet. RoboCasa fournit des démonstrations, mais leur utilisation aurait :
- Introduit un biais de distribution difficile à évaluer
- Masqué les différences entre SAC et PPO
- Complexifié la pipeline au-delà de la portée du projet

Cette option est identifiée comme une perspective d'amélioration (section 14).

### 6.4 Curriculum learning

Le curriculum learning n'a pas été implémenté. La tâche a été présentée dès le départ avec sa complexité complète. Une approche curriculum pourrait démarrer avec un angle de porte déjà partiellement ouvert et augmenter progressivement la difficulté — identifié comme perspective.

---

## 7. Reward shaping

### 7.1 Problème avec la reward naive

La reward native de RoboCasa pour `OpenCabinet` est une combinaison de :
1. Reward d'approche : bonus proportionnel à la proximité EEF → poignée
2. Reward de succès : bonus sparse si θ ≥ seuil

Ce design naïf conduit systématiquement à du **reward hacking** :

- L'agent apprend à **rester immobile devant la poignée** (*hover hacking*) : il accumule la reward d'approche sans jamais ouvrir la porte.
- L'agent peut **osciller la porte** pour accumuler des micro-rewards d'approche.
- L'agent peut **stagner près de la poignée** sans jamais tenter d'ouvrir.

Ces comportements produisent une reward cumulative élevée (~200–400) mais un taux de succès nul.

### 7.2 Reward façonnée anti-hacking

La formule complète implementée dans `robocasa_telecom/envs/reward.py` :

```
R_t = w_approach  × r_approach(d_t)        [guidage initial]
    + w_progress  × r_progress(θ_t)         [signal principal]
    + w_success   × r_success(θ_t)          [objectif final]
    - w_action_reg × ‖a_t‖²                 [régularisation]
    - w_stagnation × p_stag(d_t, ctr_t)     [anti-hover]
    - w_wrong_dir  × p_back(θ_t, θ_best)    [anti-recul]
    - w_oscillation × p_osc(Δθ_window)      [anti-oscillation]
```

**Coefficients :**

| Composante | w | Valeur r/p | Condition |
|---|---:|---|---|
| approach | 0.05 | clip(1 − d/d_max, 0, 1) | θ < 0.90 seulement |
| progress | 1.0 | max(0, θ − θ_best) / θ_success | Haute-watermark |
| success | 5.0 | 1 si θ ≥ 0.90 | Sparse |
| action_reg | 0.01 | −‖a‖² | Toujours |
| stagnation | 0.05 | −1 | d < 0.12 m ET ctr ≥ 50 steps |
| wrong_dir | 0.3 | −(θ_best − θ) | θ < θ_best − 0.02 |
| oscillation | 0.2 | −sign_changes/window | ≥ 4 changements sur 20 steps |

### 7.3 Mécanismes clés

**High-watermark progress :** `r_progress = max(0, θ - θ_best)`. L'agent ne reçoit de reward de progression que s'il bat son meilleur angle de porte de l'épisode. Impossible de gagner en oscillant ou en reculant.

**Gating de l'approche :** `r_approach = 0` dès que θ ≥ 0.90. L'agent ne peut pas continuer à accumuler la reward d'approche une fois la tâche accomplie.

**Stagnation conditionnelle :** la pénalité ne s'active que si (a) le robot est déjà proche de la poignée (d < 0.12 m) ET (b) il n'a pas progressé depuis 50 steps consécutifs. Évite les faux positifs lors de la phase d'exploration initiale.

**Oscillation sur fenêtre glissante :** un buffer circulaire de 20 steps mémorise les signes des Δθ. Si ≥ 4 changements de signe sont détectés sans progression nette, la pénalité s'active.

### 7.4 Métriques de vérification anti-hacking

| Signal | Valeur saine | Signe d'alerte |
|---|---|---|
| `approach_frac_mean` | < 0.3 | > 0.5 → hover hacking |
| `stagnation_steps_mean` | < 20 | > 100 → blocage fréquent |
| `sign_changes_mean` | < 5 | > 15 → oscillation |
| `door_angle_max_mean` | > 0.5 | < 0.2 → agent bloqué loin |
| `reward_without_success` | Faible | Élevé → gaming intermédiaire |

---

## 8. Configuration expérimentale

### Environnement

```yaml
task: OpenCabinet
robots: PandaOmron
horizon: 500
control_freq: 10
use_camera_obs: false      # observations d'état uniquement
obj_registries: [objaverse]
```

### SAC (run principal)

```yaml
total_timesteps: 3_000_000
n_envs: 12                 # SubprocVecEnv
learning_rate: 3.0e-4
buffer_size: 1_000_000
batch_size: 512
gradient_steps: 12
ent_coef: auto
net_arch: [256, 256]
eval_freq: 25_000          # validation tous les 25k steps
n_eval_episodes: 50
early_stopping_patience: 20
```

### PPO (baseline)

```yaml
total_timesteps: 5_000_000
learning_rate: 3.0e-4
n_steps: 1024
batch_size: 256
n_epochs: 10
gae_lambda: 0.95
clip_range: 0.2
net_arch: [256, 256]
eval_freq: 25_000
n_eval_episodes: 50
```

---

## 9. Métriques suivies

### Performance

- **`val_success_rate`** : taux de succès sur le split validation (seed=10000). Métrique principale.
- **`val_return_mean`** : return moyen par épisode. Indicateur de progrès général.
- **`val_door_angle_final_mean`** : angle de porte normalisé à la fin de l'épisode.
- **`val_door_angle_max_mean`** : meilleur angle atteint (high-watermark). Révèle si l'agent approche de la solution sans finir.

### Anti-hacking

- **`val_approach_frac_mean`** : part de la reward d'approche. > 0.5 = hover hacking.
- **`val_stagnation_steps_mean`** : steps passés proche de la poignée sans progrès.
- **`val_sign_changes_mean`** : changements de signe de Δθ par épisode (oscillation).
- **`reward_without_success`** : return moyen des épisodes en échec. Élevé = gaming possible.

### Qualité des actions

- **`val_action_smoothness_mean`** : `mean(‖a_t − a_{t-1}‖)`. Mesure la jerkiness.
- **`val_action_magnitude_mean`** : norme moyenne des actions.

### Algorithme

- **`train/entropy_loss`**, **`train/actor_loss`**, **`train/critic_loss`** (SAC)
- **`train/approx_kl`**, **`train/clip_fraction`** (PPO)
- **`train/fps`** : steps/seconde (indicateur de performance computationnelle)

---

## 10. Protocole expérimental

### Runs planifiés

| # | Run | Algo | Steps | Workers | Seeds | But |
|---|---|---|---:|---:|---|---|
| 0 | SAC debug | SAC | 300k | 12 | 0 | Validation reward shaping |
| 1 | SAC principal | SAC | 3M | 12 | 0 | Référence principale |
| 2 | SAC tuned | SAC | 2M | 12 | 0 | Variante hyperparamètres |
| 3 | PPO baseline | PPO | 5M | 1 | 0 | Baseline comparative |

### Protocole d'évaluation

- **Split validation** (seed=10000) : évalué tous les 25k steps pendant l'entraînement.
- **Split test** (seed=20000) : évalué une seule fois sur `best_model.zip` à la fin.
- **Nombre d'épisodes** : 50 pour la validation périodique, 50 pour l'évaluation finale.
- **Politique déterministe** : utilisée à l'évaluation (argmax de l'acteur).

### Critère de sélection du meilleur checkpoint

Le checkpoint sauvegardé comme `best_model.zip` est celui qui maximise `val_success_rate`. En cas d'égalité, `val_return_mean` sert de tiebreaker.

### Critères de succès du projet

| Cible | Condition |
|---|---|
| Val. success > 80 % | SAC principal (3M) |
| Écart train/val < 20 % | Pas de surentraînement sévère |
| `approach_frac` < 0.3 | Pas de hover hacking |
| Test success > 60 % | Généralisation aux seeds non vus |

---

## 11. Résultats

> **Note** : cette section sera complétée avec les valeurs issues des runs finaux. Les placeholders `[À compléter]` seront remplacés après exécution complète des expériences.

### 11.1 Tableau récapitulatif

| Run | Algo | Steps | Val. Success | Test Success | Best Step | `approach_frac` |
|---|---|---:|---:|---:|---:|---:|
| SAC debug | SAC | 300k | [À compléter] | — | [À compléter] | [À compléter] |
| SAC principal | SAC | 3M | [À compléter] | [À compléter] | [À compléter] | [À compléter] |
| SAC tuned | SAC | 2M | [À compléter] | [À compléter] | [À compléter] | [À compléter] |
| PPO baseline | PPO | 5M | [À compléter] | [À compléter] | [À compléter] | [À compléter] |

### 11.2 Courbes d'apprentissage

> Courbes générées avec `make plot PLOT_RUNS="outputs/OpenCabinet_SAC_seed0_*/"`.
> Fichiers : `outputs/plots/success_rate.png`, `summary.png`, `door_angle.png`, `anti_hacking.png`.

[Insérer `outputs/plots/summary.png` ici après le run final]

### 11.3 Analyse du comportement (vidéo)

> Généré avec `make eval-video CONFIG=... CHECKPOINT=... EPISODES=20`.
> Fichier : `outputs/eval/videos/<run_id>_best_episode_<ts>.mp4`.

[Décrire le comportement observé dans la vidéo du meilleur épisode]

### 11.4 Comparaison best vs final

| Run | Success @best | Success @final | Écart | Interprétation |
|---|---:|---:|---|---|
| SAC principal | [À compléter] | [À compléter] | [À compléter] | [À compléter] |
| PPO baseline | [À compléter] | [À compléter] | [À compléter] | [À compléter] |

---

## 12. Discussion

### 12.1 SAC vs PPO

[À compléter après les runs — analyser : convergence, stabilité, succès final, coût computationnel]

Éléments d'analyse attendus :
- SAC devrait converger plus vite (off-policy, 12 workers, gradient_steps=12)
- PPO devrait être plus stable mais nécessiter plus de steps
- L'écart de performance final révèle si la sample efficiency de SAC compense sa complexité de tuning

### 12.2 Efficacité du reward shaping

[À compléter — comparer `approach_frac` avant/après le reward shaping via SAC debug]

Indicateurs à analyser :
- Évolution de `approach_frac` au fil du training
- Présence ou absence d'oscillation (`sign_changes_mean`)
- Le robot atteint-il des `door_angle_max` > 0.9 avant la convergence ?

### 12.3 Surentraînement

[À compléter — comparer train success vs validation success en fin de run]

### 12.4 Généralisation

[À compléter — comparer validation success (seed=10000) vs test success (seed=20000)]

---

## 13. Difficultés rencontrées et solutions

### 13.1 Instabilité API du GymWrapper

**Problème :** le `GymWrapper` natif de RoboSuite retourne parfois une shape d'observation différente entre deux appels à `reset()`, ce qui casse les buffers de Stable-Baselines3.

**Solution :** implémentation d'un `RawRoboCasaAdapter` custom dans `envs/factory.py` qui flattèn le dictionnaire d'observations via `gymnasium.spaces.utils.flatten`, garantissant une shape constante.

### 13.2 Reward hacking (hover hacking)

**Problème :** avec la reward naive, l'agent converge vers un comportement de stationnement devant la poignée (approach_frac → 1.0, success_rate → 0 %).

**Solution :** reward shaping anti-hacking complet avec high-watermark progress, gating de l'approche, pénalités conditionnelles (section 7).

### 13.3 Crash de reprise automatique (boucle infinie)

**Problème :** un run qui crashait après la sauvegarde de `final_model.zip` mais avant `train_summary.json` était détecté comme incomplet et relancé indéfiniment.

**Solution :** `checkpoints.py` — `_run_is_complete()` vérifie désormais `final_model.json` comme fallback avec `num_timesteps >= target_total_timesteps`.

### 13.4 Incompatibilité `VecEnv.reset(seed=n)`

**Problème :** l'évaluation finale appelait `env.reset(seed=n)` sur un `VecMonitor` qui ne supporte pas le keyword `seed`.

**Solution :** `metrics.py` détecte `isinstance(env, VecEnv)` et utilise `env.seed(n); env.reset()` à la place. L'évaluation finale utilise également un environnement single-worker dédié.

### 13.5 Rendu vidéo impossible pendant l'entraînement

**Problème :** `SubprocVecEnv` spawne les workers dans des processus séparés — le rendu OpenGL est impossible depuis ces processus.

**Solution :** approche two-pass dans `eval_video.py` — pass 1 score les épisodes sans rendu, pass 2 rejoue le meilleur épisode avec le même seed dans un environnement dédié avec rendu offscreen.

---

## 14. Limites et perspectives

### Limites actuelles

| Limite | Impact | Piste |
|---|---|---|
| Observations d'état uniquement | Pas de généralisation visuelle | Ajouter CNN + observations RGB (DrQ-v2) |
| Navigation exclue | Tâche simplifiée par rapport au robot réel | Intégrer base mobile (défi de recherche) |
| Seed unique (seed=0) | Variance inconnue entre runs | Multi-seeds (3–5 seeds) + intervalles de confiance |
| Pas de sim-to-real | Résultats valides en simulation uniquement | Domain randomization + calibration |
| Assets Objaverse fixes | Généralisation limitée aux objets vus | Augmentation procédurale plus agressive |
| Pas d'imitation learning | Convergence plus lente | Pré-entraînement BC + fine-tuning SAC |

### Perspectives d'amélioration

1. **Multi-seeds** : lancer 3–5 seeds pour chaque algo et reporter les intervalles de confiance Wilson à 95 %.
2. **Curriculum learning** : démarrer avec θ_initial = 0.5, augmenter progressivement la difficulté.
3. **Observations visuelles** : intégrer les caméras RGB avec un encodeur CNN partagé (DrQ-v2 ou SAC + ResNet).
4. **Imitation learning** : pré-entraîner par Behavior Cloning sur les démos RoboCasa, puis fine-tuner avec SAC.
5. **Normalisation d'observations** : `VecNormalize` pour stabiliser l'entraînement SAC.
6. **Hyperparameter search** : Optuna ou grid search sur lr, buffer_size, gradient_steps.

---

## 15. Conclusion

Ce projet a démontré la faisabilité d'entraîner un agent RL à ouvrir une porte de placard dans RoboCasa en partant de zéro, avec SAC comme méthode principale et une reward façonnée pour prévenir le reward hacking.

Les contributions techniques clés sont :

1. **Un système de reward anti-hacking** complet avec high-watermark progress, gating de l'approche, pénalités conditionnelles d'oscillation et de stagnation.
2. **Un pipeline d'évaluation robuste** avec splits validation/test séparés, early stopping, et métriques anti-hacking tracées dans MLflow.
3. **Un système de génération vidéo two-pass** reproductible, contournant les limitations de rendu des workers parallèles.
4. **Un mécanisme d'auto-reprise** des runs interrompus, permettant de gérer les crashs sans perdre la progression.

[Compléter avec l'analyse des résultats finaux et la réponse à la question de recherche initiale.]

---

## 16. Références

- Nasiriany, S. et al. (2024). *RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots*. arXiv:2406.02523.
- Haarnoja, T. et al. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*. ICML 2018.
- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Zhu, Y. et al. (2020). *RoboSuite: A Modular Simulation Framework and Benchmark for Robot Learning*. arXiv:2009.12293.
- Todorov, E. et al. (2012). *MuJoCo: A physics engine for model-based control*. IROS 2012.
- Raffin, A. et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations*. JMLR 22(268):1−8.

---

## 17. Justification of Methods and Tools

This section explains *why* each methodological and technical choice was made, linking each decision to the concepts covered in the IA705 — Learning for Robotics course.

---

### 17.1 Why RoboCasa?

RoboCasa (Nasiriany et al., 2024) was selected because it directly corresponds to the learning framework taught in the course: an **agent interacts with an environment**, receives **observations** and **rewards**, and learns a **policy** through trial and error.

More specifically:

- **Realistic simulation** — RoboCasa uses MuJoCo physics, which provides accurate contact dynamics essential for manipulation tasks. The kitchen environment involves real physical constraints (door hinges, handle contacts) that the agent must discover.
- **Project requirement** — the course project explicitly specifies an atomic task within RoboCasa.
- **Safe data collection** — simulation allows millions of interactions without risk of hardware damage, which is critical for RL where the agent initially behaves randomly.
- **Compatibility with continuous control** — MuJoCo handles continuous joint-torque and end-effector control spaces naturally, which is required for the 7-DoF Panda arm.
- **Procedural variety** — RoboCasa generates diverse kitchen scenes with varied objects and layouts, allowing future generalisation studies.

The alternative would have been to use a real robot, but data collection speed (500 steps × 3 million = 1.5 billion simulated seconds) would be physically impossible in a real-world setting within the project timeline.

---

### 17.2 Why PandaOmron?

The PandaOmron robot was specified in the project guidelines. It combines:

- **Franka Panda arm** (7 DoF): high-precision manipulation, compliant torque control, sub-millimeter repeatability.
- **Omron LD-250 mobile base**: omnidirectional navigation.

**Navigation is explicitly excluded** in this project. The base remains fixed, positioned facing the cabinet at reset. This choice is justified by:

1. **Scope limitation** — combining navigation and manipulation doubles the problem complexity and is an open research challenge beyond this project's scope.
2. **Task decomposition** — the course emphasises decomposing complex tasks into atomic subtasks. "Open door" is the atomic task; navigation to the door would be a separate subtask.
3. **Action space reduction** — excluding the mobile base reduces the action space from ~10 DoF to ~7 DoF, making exploration more tractable with limited compute.

The arm's 7-DoF continuous action space directly motivates the choice of actor-critic algorithms (SAC, PPO) over discrete-action methods like DQN, which cannot handle continuous high-dimensional actions efficiently.

---

### 17.3 Why an Atomic Task: Open Door?

The `OpenCabinet` task was chosen for the following pedagogical and technical reasons:

- **Tractable complexity** — the task is solvable in a few million steps with a well-tuned reward, making it appropriate for a time-constrained project.
- **Illustrative of RL challenges** — the task exhibits the core difficulties studied in the course:
  - sparse reward (the door only opens after successful contact and pulling)
  - credit assignment (which action in the 500-step episode caused the door to open?)
  - exploration (the robot must discover the correct contact strategy from random actions)
  - reward hacking (the robot can exploit intermediate rewards without solving the task)
- **Physical progression** — the door angle θ ∈ [0, 1] provides a continuous measure of task progress, enabling shaped reward design.
- **Clear evaluation** — success is unambiguous: `θ ≥ 0.90`. This allows rigorous comparison across runs and algorithms.
- **Visual interpretability** — a video immediately reveals whether the robot opens the door or exploits a degenerate strategy, making qualitative evaluation straightforward.

---

### 17.4 Link with Reinforcement Learning Course

#### Markov Decision Process (MDP)

The door-opening task is naturally formulated as an MDP:

- **States S** — joint positions and velocities, end-effector pose, distance to handle, door angle θ. The Markov property holds approximately: given the full state, the next state depends only on the current state and action.
- **Actions A** — differential end-effector control (7-dimensional continuous vector). Each action step corresponds to 0.1 seconds of simulated time (control_freq=10 Hz).
- **Reward R(s,a)** — shaped signal encouraging approach, progress, and success (see Section 7).
- **Transition T(s'|s,a)** — determined by MuJoCo physics (deterministic given state and action).
- **Discount factor γ = 0.99** — the agent values future rewards almost as much as immediate ones, which is necessary because the success reward only arrives hundreds of steps after the first useful action.

#### Exploration vs. Exploitation

The door-opening task has a severe **exploration problem**: from a random policy, the robot almost never accidentally opens a door. SAC addresses this via **entropy maximisation** — the policy is trained to simultaneously maximise return AND entropy, preventing premature convergence to a local optimum (e.g., hovering near the handle). PPO addresses it via a fixed entropy bonus (`ent_coef=0.01`) and stochastic rollouts.

#### Credit Assignment

When the door opens at step 350 of a 500-step episode, which of the preceding 350 actions caused it? This is the **credit assignment problem**. Both SAC and PPO use the **advantage function** (actor-critic) to assign credit:

- SAC uses two Q-networks (critics) to estimate the value of each (state, action) pair.
- PPO uses Generalised Advantage Estimation (GAE, λ=0.95) to reduce variance while maintaining some bias.

The discount factor γ=0.99 ensures that the reward at step 350 still propagates meaningful gradient back to earlier steps.

#### Why Not DQN or Q-Learning?

DQN operates on **discrete action spaces**. The Panda arm requires a 7-dimensional continuous action vector. Discretising each dimension into even 10 bins would produce 10^7 = 10 million discrete actions — intractable for tabular methods and impractical for neural networks.

**SAC and PPO directly parameterise continuous Gaussian policies**, making them the natural choices for continuous robotic control.

#### SAC — Soft Actor-Critic (main algorithm)

SAC (Haarnoja et al., 2018) is an **off-policy actor-critic** algorithm that maximises:

```
J(π) = Σ_t E[R(s_t, a_t) + α · H(π(·|s_t))]
```

Key features relevant to this project:

| Feature | Course concept | Practical impact |
|---|---|---|
| Off-policy learning | Replay buffer → data reuse | 3M steps instead of 10M+ |
| Entropy regularisation | Exploration vs exploitation | No manual exploration schedule |
| Two critics (min) | Overestimation bias reduction | Stable Q-value estimates |
| Automatic α tuning | Hyperparameter sensitivity | Less manual tuning |
| 12 parallel workers | Sample collection speed | ~2 600 steps/min |

**SAC metrics to monitor in MLflow:**
- `train/actor_loss` — should decrease (policy improvement)
- `train/critic_loss` — should converge (Q-function learning)
- `train/entropy_loss` — tracks entropy evolution
- `train/ent_coef` — should stabilise around a fixed value
- `train/q_values` — should increase as the policy improves

#### PPO — Proximal Policy Optimization (baseline)

PPO (Schulman et al., 2017) is an **on-policy actor-critic** algorithm with clipped surrogate objective:

```
L_CLIP(θ) = E[min(r_t(θ)·Â_t, clip(r_t(θ), 1−ε, 1+ε)·Â_t)]
```

Key features relevant to this project:

| Feature | Course concept | Practical impact |
|---|---|---|
| Clipped updates | Policy update stability | Avoids catastrophic forgetting |
| GAE (λ=0.95) | Advantage estimation, bias-variance | Smooth credit assignment |
| On-policy rollouts | Fresh data from current policy | No stale experience issue |
| n_steps=1024 | Rollout horizon before update | Balances update frequency |
| `n_epochs=10` | Data reuse within update | Efficient on-policy learning |

**PPO metrics to monitor in MLflow:**
- `train/approx_kl` — should remain < 0.05 (if > 0.1, update is too large)
- `train/clip_fraction` — should remain < 0.3 (if > 0.5, clip_range is too small)
- `train/explained_variance` — should approach 1.0 (value function accuracy)
- `train/entropy_loss` — slightly negative, encouraging exploration

**Why PPO as baseline rather than main algorithm?** PPO is on-policy: it discards all collected experience after each update. With a 500-step horizon and 12 workers, each PPO update uses ~12,000 fresh steps, then throws them away. SAC, with its 1M-transition replay buffer, reuses each experience ~12 times (gradient_steps=12), making it far more sample-efficient for this type of task.

---

### 17.5 Why Reward Shaping?

#### The Sparse Reward Problem

A naive reward function `R = 1 if θ ≥ 0.90 else 0` creates an **extreme exploration problem**. Starting from random actions, the probability of a 7-DoF arm randomly performing the exact sequence of actions to open a cabinet door is negligibly small. In practice, training with a purely sparse reward on this task results in zero successful episodes for hundreds of thousands of steps — the agent never receives a learning signal.

This connects directly to the course concept of **feedback delay**: the agent must perform ~100–200 meaningful actions before receiving any reward, making credit assignment nearly impossible without guidance.

#### The Shaped Reward

Dense intermediate signals are added following the **potential-based reward shaping** principle (Ng et al., 1999), which guarantees that the optimal policy is preserved:

1. **Approach reward** — guides the agent toward the handle before it knows how to interact with it. Essential in the early training stages.
2. **Progress reward (high-watermark)** — rewards new progress beyond the best angle ever reached. This is the main learning signal.
3. **Success bonus** — sparse dominant signal ensuring the agent always prefers task completion.

#### The Reward Hacking Problem

Reward shaping introduces risk: the agent may find ways to maximise the shaped reward *without* solving the task. Observed instances in this project:

- **Hover hacking** — the agent parks 5 cm from the handle to accumulate approach reward. Result: return ~200–400, success rate = 0%.
- **Oscillation** — the agent rapidly oscillates the door without net progress, exploiting poorly-designed progress signals.

These correspond to the course concept of **reward hacking / Goodhart's law**: when a measure becomes a target, it ceases to be a good measure.

#### Anti-Hacking Guards

The anti-hacking reward design (see `docs/reward_shaping.md`) addresses each failure mode:

- **High-watermark progress** — oscillation gives zero reward because `r_progress = max(0, θ − θ_best)`.
- **Approach gating** — approach reward disabled once the door is open (θ ≥ 0.90), preventing hover farming after success.
- **Stagnation penalty** — triggered only when the robot is already near the handle (d < 0.12 m) AND hasn't progressed for 50+ steps, avoiding false positives during early exploration.
- **Success dominance** — `w_success × r_success` is 100× larger than the maximum possible approach reward over an episode.

---

### 17.6 Why Curriculum Learning?

Curriculum learning (Bengio et al., 2009) was **not implemented** in the current version of this project. However, it is identified as a key improvement direction.

The core idea — starting with an easier version of the task and progressively increasing difficulty — directly addresses the exploration problem described above. For the door-opening task, a curriculum might look like:

- **Stage 1** — initialise the door at θ=0.5 (already half-open); the agent only needs to push it the remaining 40%.
- **Stage 2** — initialise at θ=0.2; full task but with a head start.
- **Stage 3** — full task with random initialisation θ=0.

This would drastically reduce the number of steps before the first success, accelerating credit assignment learning. The course module on curriculum learning and multi-task learning motivates this approach as a principled solution to sparse-reward exploration.

---

### 17.7 Why Not Imitation Learning?

Imitation learning (BC, GAIL, IQL) was **not used** in this project, but RoboCasa provides human demonstrations for the `OpenCabinet` task.

The reason for not using them:

1. **Comparative clarity** — using demonstrations would mix two learning paradigms, making it harder to attribute performance differences between SAC and PPO.
2. **Project scope** — the course focuses on RL; imitation learning would shift the focus away from the core assignment.
3. **Timeline** — integrating demonstrations would require additional preprocessing and validation.

If imitation learning were used, the recommended approach would be:
- **Behavioral Cloning (BC)** pre-training — train the policy network on demonstrations with supervised learning.
- **Fine-tuning with SAC** — use the BC-initialised policy as a warm start for RL, avoiding the random exploration phase entirely.

The main risk of pure BC is **covariate shift**: the policy encounters states not seen in the demonstrations and has no guidance for how to behave. RL fine-tuning corrects this.

---

### 17.8 Why MLflow?

MLflow was chosen as the experiment tracking backend for the following reasons:

1. **Experiment reproducibility** — every run logs its exact hyperparameters, config files, and resolved YAML. Any run can be exactly identified and reproduced.
2. **Metric comparison** — MLflow's comparison UI allows side-by-side plots of `val_success_rate` across SAC, SAC tuned, and PPO runs.
3. **Artifact storage** — checkpoints, videos, and CSV files are stored as MLflow artifacts, making it easy to retrieve the exact model that produced a given result.
4. **RL instability diagnosis** — the course emphasises that RL algorithms are inherently unstable. MLflow makes it possible to detect instability patterns (e.g., sudden drops in success rate, reward spikes) and correlate them with specific training events.
5. **Professional workflow** — MLflow is an industry-standard tool (used at Databricks, Meta, etc.) that demonstrates research engineering maturity.

To launch the MLflow UI:
```bash
uv run mlflow ui --backend-store-uri ./mlruns
# Then open http://127.0.0.1:5000
```

---

### 17.9 Why 12 Parallel Workers?

The choice of `n_envs=12` (SubprocVecEnv) balances several constraints:

**Benefits:**
- **Data throughput** — 12 workers collect 12 episodes in parallel, multiplying experience collection speed by ~12×.
- **Trajectory diversity** — different workers encounter different random initialisations, reducing correlation in the replay buffer (crucial for off-policy methods like SAC).
- **Wall-clock efficiency** — on a 12-core CPU + MPS GPU system, 12 workers saturate available parallelism.

**Costs and limitations:**
- **RAM** — each worker loads a full RoboCasa simulation (~460 MB). 12 workers = ~5.5 GB just for workers.
- **Render impossibility** — `SubprocVecEnv` uses `spawn`/`fork` to create worker processes; OpenGL/MuJoCo rendering contexts cannot be shared across processes. This is why video generation must be done post-training with a dedicated single-worker environment.
- **Debug complexity** — associating an episode to a specific worker requires careful seed management (each worker uses `base_seed + worker_id`).
- **On-policy caveat** — for PPO (on-policy), 12 workers collect rollouts that are all used in the *same* update batch, then discarded. This is fine because the data is fresh. For SAC (off-policy), the replay buffer collects from all 12 workers independently, which is beneficial.

**Why debug runs first?** The 300k-step SAC debug run exists precisely to validate that:
- the reward shaping does not cause hover hacking
- the 12-worker setup does not cause memory issues
- MLflow logging works correctly
- the validation callback fires at the right frequency

Only after a clean debug run is it safe to launch the expensive 3M-step run.

---

### 17.10 Why Video Generation?

Numerical metrics alone are insufficient to validate that a robot RL agent has learned the *correct* behaviour. This is a well-known problem in robot learning, sometimes called the **metric-behaviour gap**:

- An agent with `return_mean = 320` might be successfully opening the door — or it might be hovering at 3 cm from the handle for 490 steps, collecting approach reward.
- `door_angle_final_mean = 0.91` confirms the door is opened, but not *how* — the robot might be pushing from an awkward angle that would fail on a real robot.

Video generation provides **qualitative validation** that:
1. The robot approaches the handle in a sensible way.
2. The robot makes contact and pulls/pushes the door open.
3. There is no degenerate behaviour (oscillation, spinning, collision farming).

For the course submission, a video of a successful episode is required evidence that the agent has genuinely learned the task.

The **two-pass approach** (see `docs/video_generation.md`) was necessary because rendering is impossible during SubprocVecEnv training. Pass 1 scores N episodes without rendering; pass 2 re-renders the best episode with the same seed, ensuring reproducibility.

---

### 17.11 Why These Metrics?

| Metric | Justification |
|---|---|
| `val_success_rate` | The *only* metric that directly answers "did the agent complete the task?" — not a proxy, not a surrogate |
| `val_door_angle_final_mean` | Physical measurement of actual door opening; detects near-misses (θ=0.85 vs θ=0.00) |
| `val_door_angle_max_mean` | High-watermark: detects oscillation (max >> final means the robot opens then closes) |
| `val_approach_frac_mean` | Reward hacking detector: > 0.5 means hover hacking is dominant |
| `val_stagnation_steps_mean` | Measures time stuck near handle without progress |
| `val_sign_changes_mean` | Oscillation counter: high = door moved back-and-forth without net progress |
| `reward_without_success` | If this is high and success_rate is low, the reward is being gamed |
| `val_action_smoothness_mean` | Detects jerky, unstable policies (potential sim-to-real transfer risk) |
| `episode_length_mean` | Decreasing = agent solves faster; constant at 500 = never succeeds |
| `val_success_ci_lo / ci_hi` | 95% Wilson confidence interval — quantifies statistical uncertainty over N=50 episodes |

Using `success_rate` as the primary metric, rather than `return_mean`, is a deliberate design decision aligned with the course: **the reward function is a means to an end, not the end itself**. Optimising return without monitoring task success leads directly to reward hacking.

---

### 17.12 Why Not Only Reward?

The total episodic return is a **necessary but not sufficient** condition for task success. An agent can maximise return through several degenerate strategies:

- Hover hacking (approach reward farming)
- Micro-oscillation (exploiting poorly-designed progress signals)
- Collision bouncing (if reward for door contact is not gated)

The course concept of **Goodhart's Law** applies: *when a measure becomes a target, it ceases to be a good measure*. The shaped reward is a measure designed to guide the agent toward task success; it is not itself the success criterion.

Therefore, this project always reports *both*:
1. `val_success_rate` — the true task criterion (door open ≥ 90%)
2. `val_return_mean` — the proxy signal used for learning

And monitors anti-hacking metrics (`approach_frac`, `stagnation_steps`, `sign_changes`) to detect when the proxy diverges from the true criterion.

---

## 18. Relation to Course Concepts

The table below maps every major concept from the IA705 — Learning for Robotics course to its concrete use in this project.

| Course concept | Use in this project |
|---|---|
| **Markov Decision Process (MDP)** | State = joint positions + EEF pose + door angle; Action = 7D continuous arm command; Reward = shaped anti-hacking signal; γ = 0.99; T = 500 steps |
| **Policy π(a\|s)** | Stochastic Gaussian MLP [256, 256] trained via SAC (main) and PPO (baseline); deterministic at evaluation |
| **Value function V(s) / Q(s,a)** | Two Q-networks (SAC critics) estimating cumulative discounted return; advantage function in PPO via GAE |
| **Reward signal** | Dense (approach, progress) + sparse (success bonus); weighted sum with anti-hacking guards |
| **Exploration vs. exploitation** | SAC: entropy maximisation (automatic α); PPO: entropy bonus (ent_coef=0.01) + stochastic rollouts |
| **Credit assignment** | Advantage function (PPO/GAE λ=0.95) and Q-function (SAC) attribute reward to actions 100–300 steps earlier |
| **Reward shaping** | 7-component shaped reward (approach, progress, success, action_reg, stagnation, wrong_dir, oscillation) |
| **Reward hacking / Goodhart's law** | Observed: hover hacking (approach_frac > 0.5, success_rate = 0); solved via high-watermark progress and gating |
| **Feedback delay** | Success reward arrives at step ~150–350; γ=0.99 and advantage estimation propagate this backward |
| **Actor-Critic architecture** | Both SAC and PPO use shared actor-critic networks [256, 256]; critic guides actor updates |
| **Off-policy learning (SAC)** | Replay buffer (1M transitions) allows reuse of past experience; crucial for sample efficiency |
| **On-policy learning (PPO)** | Fresh rollouts at each update; more stable but less sample-efficient |
| **Entropy regularisation** | SAC maximises H(π(·\|s)) at each step; prevents premature convergence to deterministic sub-optimal policy |
| **Continuous action spaces** | DQN/tabular Q-learning inapplicable; Gaussian policy + reparameterisation trick required |
| **Curriculum learning** | Not implemented; identified as key future improvement to address sparse-reward exploration |
| **Imitation learning** | Not used; identified as future improvement (BC pre-training + SAC fine-tuning) |
| **Evaluation over multiple seeds** | Single seed (seed=0) due to time constraints; variance analysis identified as limitation |
| **Validation / test split** | Training uses seed=0; validation uses seed=10000; test uses seed=20000 — unseen configurations |
| **Overfitting in RL** | Monitored via train vs. validation success gap; `best_model.zip` saved at validation peak, not training end |
| **Visual evaluation** | Two-pass video generation confirms qualitative task completion beyond numerical metrics |
| **Parallelisation** | 12 SubprocVecEnv workers; each uses a different episode seed for trajectory diversity |
| **Reproducibility** | Fixed seeds, pinned library commits, saved configs, replay buffer checkpointing |
