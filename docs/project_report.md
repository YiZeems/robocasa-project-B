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
