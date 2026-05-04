# Prompt-guide projet RL RoboCasa — Tâche atomique : ouvrir une porte

## Objectif du projet

Ce projet vise à entraîner un agent de Reinforcement Learning dans RoboCasa / RoboCasa-like environment sur une tâche atomique de manipulation robotique : **ouvrir une porte**.

L'objectif n'est pas seulement d'obtenir un bon reward en entraînement, mais d'obtenir une politique robuste capable de généraliser à plusieurs seeds, positions initiales, layouts et variations d'environnement.

---

## Question centrale du projet

> Dans un environnement basé sur RoboCasa, quelle méthode d'apprentissage est la plus adaptée pour entraîner une tâche atomique comme ouvrir une porte : PPO, SAC, combinaison PPO/SAC, ou une autre méthode plus performante ?
> À partir de quand l'entraînement devient-il du surentraînement ?
> Quels hyperparamètres et critères d'évaluation utiliser ?

---

# 1. Méthodes à comparer

## Méthode recommandée en priorité

La meilleure approche pratique est généralement :

```text
Démonstrations RoboCasa
→ Behavior Cloning / Diffusion Policy
→ Fine-tuning RL avec SAC
→ Sélection du meilleur checkpoint sur validation success rate
```

Cette approche est préférable lorsque des démonstrations sont disponibles, car la tâche "ouvrir une porte" exige une séquence structurée :

1. Approcher la poignée.
2. Aligner le gripper.
3. Saisir la poignée.
4. Tirer ou pousser selon le mécanisme.
5. Maintenir la trajectoire jusqu'à l'ouverture suffisante.

Le RL pur peut prendre beaucoup de temps à découvrir spontanément cette séquence.

---

## Classement recommandé des méthodes

| Rang | Méthode | Cas d'usage | Verdict |
|---:|---|---|---|
| 1 | Démonstrations + RL fine-tuning | Si des démonstrations existent | Meilleur choix pratique |
| 2 | SAC | Contrôle continu, reward dense, observations état | Très bon baseline |
| 3 | DrQ-v2 / SAC image-based | Si observations visuelles RGB | Recommandé pour apprentissage depuis pixels |
| 4 | TD3 + HER | Reward sparse avec objectif clair | Possible mais plus fragile |
| 5 | PPO | Beaucoup d'environnements parallèles, besoin de stabilité | Bon baseline mais moins sample-efficient |
| 6 | Combinaison PPO + SAC | Expérimental | À éviter au début |

---

# 2. Choix principal recommandé

## Si observations état bas niveau

Utiliser en priorité :

```text
SAC
```

Justification :

- Algorithme off-policy.
- Meilleure sample efficiency que PPO.
- Adapté au contrôle continu.
- Replay buffer utile pour réutiliser les expériences.
- Entropy maximization utile pour l'exploration.

---

## Si observations visuelles RGB

Utiliser plutôt :

```text
DrQ-v2
ou
SAC avec encoder CNN
ou
Diffusion Policy pré-entraînée puis fine-tuning
```

Éviter de commencer avec PPO brut sur images, sauf si l'on dispose de beaucoup de compute et d'environnements parallèles.

---

## Si démonstrations disponibles

Pipeline recommandé :

```text
1. Pré-entraînement par imitation learning
2. Évaluation sur validation seeds
3. Fine-tuning SAC
4. Early stopping sur validation success rate
```

Méthodes possibles :

```text
Behavior Cloning
Diffusion Policy
Residual RL
SAC fine-tuning
```

---

# 3. Définition du surentraînement en RL

Il n'existe pas de nombre fixe de steps ou d'epochs où le surentraînement commence.

Le surentraînement apparaît lorsque :

```text
train_success_rate augmente ou reste très haut
mais
validation_success_rate ou test_success_rate stagne ou baisse
```

Le dernier checkpoint n'est donc pas forcément le meilleur.

---

## Signes typiques de surentraînement

Surentraînement probable si :

```text
train_success_rate - validation_success_rate > 20 %
```

ou si :

```text
validation_success_rate ne progresse plus pendant 10 à 20 évaluations
```

ou si :

```text
train reward continue d'augmenter
mais
test success rate baisse
```

---

## Exemple de diagnostic

| Step | Train success | Validation success | Décision |
|---:|---:|---:|---|
| 500k | 45 % | 38 % | Continuer |
| 1M | 78 % | 70 % | Continuer |
| 1.5M | 95 % | 82 % | Bon checkpoint |
| 2M | 99 % | 79 % | Début possible de surentraînement |
| 2.5M | 100 % | 72 % | Garder checkpoint 1.5M |

---

# 4. Nombre de steps recommandé

## Ordres de grandeur

| Configuration | Steps recommandés |
|---|---:|
| Sanity check simple | 10k – 100k |
| Premiers apprentissages | 100k – 500k |
| SAC avec observations état | 1M – 3M |
| PPO avec observations état | 3M – 5M |
| Apprentissage depuis images | 5M – 20M+ |
| Fine-tuning après démonstrations | 500k – 2M |

---

## Recommandation pour ce projet

Commencer avec :

```text
SAC état bas niveau : 3M steps maximum
évaluation toutes les 25k steps
early stopping si validation ne progresse plus
```

Puis comparer avec :

```text
PPO : 5M steps maximum
```

---

# 5. Reward shaping pour ouvrir une porte

La reward ne doit pas seulement être sparse, sinon l'agent risque de ne jamais découvrir la bonne séquence.

## Reward dense recommandée

```text
reward =
  + distance_reward(hand, handle)
  + grasp_reward(handle)
  + door_angle_reward
  + success_bonus
  - action_penalty
  - collision_penalty
```

---

## Exemple concret

```text
r =
  1.0 * exp(-5 * distance_to_handle)
+ 1.0 * is_grasping_handle
+ 3.0 * normalized_door_open_angle
+ 10.0 * success
- 0.01 * ||action||²
```

---

## Critère de succès

Définir le succès par un seuil clair :

```text
success = door_angle > threshold
```

ou :

```text
success = door_joint_position > threshold
```

Exemple :

```text
success = door_open_angle > 30 degrés
```

ou selon la représentation RoboCasa :

```text
success = joint_position_door > threshold_value
```

---

# 6. Configuration SAC recommandée

## SAC — observations état bas niveau

```yaml
algo: SAC
total_timesteps: 3_000_000

learning_rate: 3e-4
buffer_size: 1_000_000
batch_size: 256

gamma: 0.99
tau: 0.005

train_freq: 1
gradient_steps: 1
learning_starts: 10_000

ent_coef: auto
target_update_interval: 1

policy_network: [256, 256]
q_network: [256, 256]

normalize_observations: true
normalize_rewards: true
```

---

## SAC — si exploration difficile

```yaml
learning_starts: 25_000
batch_size: 512
buffer_size: 2_000_000
ent_coef: auto_0.2
```

---

## SAC — si apprentissage instable

Tester :

```yaml
learning_rate: 1e-4
batch_size: 512
tau: 0.005
gamma: 0.98
```

---

# 7. Configuration PPO recommandée

## PPO baseline

```yaml
algo: PPO
total_timesteps: 5_000_000

n_envs: 16
n_steps: 1024
batch_size: 256
n_epochs: 10

learning_rate: 3e-4
gamma: 0.99
gae_lambda: 0.95

clip_range: 0.2
ent_coef: 0.01
vf_coef: 0.5
max_grad_norm: 0.5

normalize_observations: true
normalize_rewards: true

policy_network: [256, 256]
value_network: [256, 256]
```

---

## PPO — variante plus stable

Si PPO diverge ou devient trop instable :

```yaml
learning_rate: 1e-4
clip_range: 0.1
ent_coef: 0.005
n_epochs: 5
```

---

# 8. Évaluation et validation

## Ne jamais juger uniquement avec le reward d'entraînement

Les métriques obligatoires sont :

| Métrique | Objectif |
|---|---:|
| Train success rate | > 90 % |
| Validation success rate | > 80 % si possible |
| Test success unseen layouts | > 60–70 % |
| Écart train/test | < 15–20 points |
| Episode length | Doit diminuer |
| Door angle final | Doit dépasser le seuil |
| Action magnitude | Doit rester raisonnable |

---

## Protocole d'évaluation

```yaml
eval_freq: 25_000
n_eval_episodes: 50
validation_seeds: 20
test_seeds: 50
checkpoint_metric: validation_success_rate
early_stopping_patience: 10
```

---

## Early stopping

Arrêter ou sauvegarder le meilleur modèle si :

```text
validation_success_rate ne s'améliore pas pendant 250k–500k steps
```

ou :

```text
train_success_rate - validation_success_rate > 20 %
```

---

# 9. Plan expérimental recommandé

## Expérience 1 — Sanity check

Objectif : vérifier que l'environnement, la reward et les actions fonctionnent.

```text
Méthode : SAC
Observations : état bas niveau
Layouts : 1
Steps : 500k
Reward : dense
```

Critère de succès :

```text
train_success_rate >= 80 %
```

Si l'agent n'apprend pas ici, il ne faut pas passer à une configuration plus complexe. Il faut d'abord corriger :

- reward,
- action space,
- observation space,
- success condition,
- reset distribution.

---

## Expérience 2 — Généralisation simple

```text
Méthode : SAC
Layouts : 10–20
Steps : 3M
Évaluation : toutes les 25k steps
```

Critère de succès :

```text
validation_success_rate >= 70 %
```

---

## Expérience 3 — PPO baseline

```text
Méthode : PPO
n_envs : 16 ou 32
Steps : 5M
```

Objectif :

```text
Comparer stabilité, reward final, success rate et sample efficiency avec SAC
```

---

## Expérience 4 — Démonstrations + RL

```text
Méthode :
Behavior Cloning ou Diffusion Policy
puis
SAC fine-tuning
```

Configuration :

```text
Pré-entraînement imitation : jusqu'à convergence validation
Fine-tuning SAC : 500k–2M steps
```

Objectif :

```text
Réduire le temps d'apprentissage
améliorer la robustesse
améliorer la réussite sur test seeds
```

---

## Expérience 5 — Vision

Si observations RGB :

```text
Méthode : DrQ-v2 ou SAC + CNN encoder
Steps : 5M–20M+
Augmentations : random crop, color jitter léger, frame stacking si utile
```

Éviter :

```text
PPO brut depuis pixels comme première méthode
```

---

# 10. Hyperparamètres à tuner en priorité

## SAC

Tuner dans cet ordre :

1. Reward shaping.
2. Learning rate.
3. Batch size.
4. Entropy coefficient.
5. Buffer size.
6. Gamma.
7. Network size.

Grille de départ :

```yaml
learning_rate: [3e-4, 1e-4]
batch_size: [256, 512]
gamma: [0.98, 0.99]
ent_coef: [auto, auto_0.2]
buffer_size: [1_000_000, 2_000_000]
```

---

## PPO

Tuner dans cet ordre :

1. Learning rate.
2. Clip range.
3. Entropy coefficient.
4. Number of environments.
5. n_steps.
6. n_epochs.

Grille de départ :

```yaml
learning_rate: [3e-4, 1e-4]
clip_range: [0.2, 0.1]
ent_coef: [0.0, 0.005, 0.01]
n_envs: [16, 32]
n_steps: [512, 1024, 2048]
n_epochs: [5, 10]
```

---

# 11. Checklist debug

Si l'agent n'apprend pas, vérifier dans cet ordre :

## Environnement

- La porte est-elle manipulable ?
- Le joint de porte est-il correctement lu ?
- Le seuil de succès est-il atteignable ?
- L'épisode dure-t-il assez longtemps ?
- Le reset place-t-il le robot dans une position réaliste ?

## Action space

- Les actions sont-elles normalisées ?
- Le gripper peut-il se fermer correctement ?
- Les vitesses sont-elles trop faibles ou trop fortes ?
- L'agent peut-il atteindre la poignée ?

## Observation space

- L'agent observe-t-il la position de la poignée ?
- L'agent observe-t-il l'angle de la porte ?
- L'agent observe-t-il l'état du gripper ?
- Les observations sont-elles normalisées ?

## Reward

- La reward augmente-t-elle quand la main approche la poignée ?
- La reward augmente-t-elle quand la porte s'ouvre ?
- Le success bonus est-il donné au bon moment ?
- L'action penalty n'est-elle pas trop forte ?

## Évaluation

- Les seeds d'évaluation sont-elles séparées des seeds d'entraînement ?
- Les layouts de test sont-ils non vus ?
- Le modèle sauvegardé est-il le meilleur checkpoint validation, pas le dernier ?

---

# 12. Décision finale recommandée

Pour ce projet, la meilleure stratégie est :

```text
1. Implémenter SAC avec observations état bas niveau.
2. Utiliser une reward dense structurée.
3. Évaluer toutes les 25k steps.
4. Sauvegarder le meilleur checkpoint selon validation success rate.
5. Comparer avec PPO comme baseline.
6. Si démonstrations disponibles, faire imitation learning puis fine-tuning SAC.
7. Si observations RGB, utiliser DrQ-v2 ou SAC avec encoder CNN plutôt que PPO brut.
```

---

# 13. Résumé très court

```text
Méthode principale : SAC
Meilleure méthode si demos : imitation learning + SAC fine-tuning
Baseline : PPO
Vision : DrQ-v2 ou SAC + CNN
Steps SAC : 1M–3M
Steps PPO : 3M–5M
Vision : 5M–20M+
Eval : toutes les 25k steps
Métrique principale : validation success rate
Surentraînement : train success haut mais validation/test stagnant ou en baisse
Checkpoint final : meilleur modèle validation, pas dernier modèle
```
