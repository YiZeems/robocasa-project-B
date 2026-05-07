# Model Improvements and Future Work

> This document identifies concrete improvements to the current project, each linked to a concept from the IA705 — Learning for Robotics course.

---

## 1. Improvements Directly Linked to Course Concepts

### Summary Table

| Amélioration | Motivation | Impact attendu | Difficulté | Priorité | Concept cours |
|---|---|---|---|---|---|
| Fix `ent_coef` : valeur fixe (v3) | Auto-tuning crash → α→0 → politique déterministe | Apprentissage stable, exploration maintenue | Faible | **Critique — fait (v3)** | Exploration vs exploitation |
| `use_sde=True` (v3) | Bruit corrélé à l'état → mouvements cohérents vers la poignée | Premier contact plus rapide | Faible | **Critique — fait (v3)** | Exploration structurée |
| `gradient_steps=4` + `tau=0.01` (v2/v3) | Critic loss spikes > 40k steps → divergence | Stabilité de l'entraînement | Faible | **Critique — fait (v2/v3)** | Actor-critic — stabilité du critic |
| LR schedule cosine (v3) | LR constant → sur-apprentissage en fin de run | Convergence plus nette, pas de remontée loss | Faible | **Haute — fait (v3)** | Optimisation |
| `eval_freq=100k` + `n_eval_episodes=10` | Validation = 82% du wall time en v1 | Throughput ×3–4 | Faible | **Haute — fait (v2/v3)** | Protocole expérimental |
| `obj_registries=[lightwheel]` | Reset 9.3s vs 4.8s avec assets légers | Throughput ×1.9 | Faible | **Haute — fait** | Efficacité computationnelle |
| `control_freq=20` | Simulation 2× plus lente que nécessaire | Throughput ×2 | Faible | **Haute — fait** | Simulation physique |
| Curriculum learning spawn + theta_success (v3_curriculum) | Cold start — exploration jamais productive sur tâche pleine | Pic à 0.021 rad à 300k, signal positif | Faible | **Haute — fait, insuffisant** | Curriculum learning |
| HER (Hindsight Experience Replay) | Buffer 100% sans succès → critic ne valorise pas l'ouverture | Premier succès virtuel dès le 1er épisode | Moyenne | **Critique — en cours** | Off-policy learning + exploration |
| Observation normalisation (`VecNormalize`) | Observations brutes sur 248D de scales très différentes | Convergence SAC plus rapide, gradients stables | Faible | Haute (post-deadline) | Variance reduction — policy gradient |
| BC pre-training + SAC fine-tuning | Exploration aléatoire gaspille des milliers d'épisodes | Succès non nul dès les premiers épisodes | Moyenne | Haute (post-deadline) | Imitation learning |
| Ablations sur les composantes reward | Impact individuel de chaque composante inconnu | Simplification du reward + compréhension | Faible–Moyenne | Haute (post-deadline) | Reward shaping — analyse |
| Multi-seed (3–5 seeds) | Seed unique → variance inconnue | Conclusions statistiquement valides | Faible (coût compute) | Haute (post-deadline) | Robustesse statistique |
| Comparaison TD3 | Isole la contribution de l'entropie (SAC vs TD3) | Compréhension de l'exploration entropique | Faible | Moyenne (post-deadline) | Off-policy actor-critic |
| Critère de succès soutenu | Succès momentané ≠ porte maintenue ouverte | Alignement évaluation / objectif réel | Faible | Moyenne (post-deadline) | Protocole d'évaluation |
| Évaluation sur configs hors distribution | Tâche validée sur les seeds d'entraînement uniquement | Test de généralisation réelle | Moyenne | Moyenne (post-deadline) | Généralisation en robotique |
| Intégration navigation (base mobile) | Base fixe = tâche simplifiée | Tâche complète PandaOmron | Très haute | Basse (post-deadline) | Manipulation + navigation |
| Sim-to-real transfer | Politique simulée ≠ comportement réel | Déploiement sur vrai robot | Très haute | Basse (post-deadline) | Sim-to-real gap |

---

### 1.1 Fix ent_coef : Valeur Fixe (v3) — Critique, Fait

**État actuel :** v1 et v2 ont échoué à cause de l'auto-tuning du coefficient
d'entropie `α`. SAC pousse systématiquement `α` vers 0 car la `log_prob` de la
politique (~−20) est structurellement inférieure au `target_entropy` (−12 ou −4).
Résultat : politique déterministe, actions saturées aux limites, `success_rate = 0%`.

**Amélioration :** `ent_coef=0.1` fixe (v3). Suppression totale de l'auto-tuning.
`α = 0.1` permanent garantit un signal d'exploration tout au long de l'entraînement.

**Lien cours :** Exploration vs exploitation dans les algorithmes maximum entropy
(SAC). Le coefficient d'entropie contrôle directement la diversité de la politique.
Un `α` trop faible rend la politique déterministe prématurément (exploitation sans
exploration suffisante).

**Impact attendu :** Stabilisation de l'entraînement, `actor_loss` qui reste
négatif, actions qui n'atteignent pas systématiquement les bornes.

---

### 1.2 State-Dependent Exploration — use_sde=True (v3)

**État actuel :** SAC utilise par défaut un bruit gaussien indépendant à chaque
step. Sur une tâche de manipulation précise, ce bruit blanc produit des mouvements
incohérents qui rendent difficile l'établissement de contact avec la poignée.

**Amélioration :** `use_sde=True` avec `sde_sample_freq=64`. Le bruit est généré
en fonction de l'état courant et maintenu cohérent pendant 64 steps (~1/8 d'épisode).
Les mouvements d'exploration sont directionnels plutôt qu'aléatoires.

**Lien cours :** Exploration structurée vs exploration aléatoire. Le SDE est un
mécanisme d'exploration adapté aux espaces d'actions continus en manipulation.

**Impact attendu :** Contact avec la poignée établi plus tôt dans l'entraînement,
réduction des épisodes entièrement gaspillés en bruit blanc.

---

### 1.3 Observation Normalisation (VecNormalize)

**État actuel :** Les observations brutes (220D) mélangent angles articulaires en
radians, distances en mètres, et vitesses — sur des échelles très différentes. SAC
reçoit des gradients d'amplitudes hétérogènes, ce qui ralentit la convergence.

**Amélioration :**

```python
from stable_baselines3.common.vec_env import VecNormalize
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
```

`VecNormalize` maintient une moyenne et un écart-type glissants pour chaque
dimension et normalise rewards à variance unitaire.

**Lien cours :** Réduction de la variance en policy gradient. La normalisation
des observations réduit la variance des gradients du critic et stabilise SAC sur
des tâches avec observations hétérogènes.

**Impact attendu :** Convergence plus rapide, critic loss plus stable, sensibilité
réduite au learning rate.

**Difficulté de mise en oeuvre :** Faible — `VecNormalize` est natif SB3. Requiert
de sauvegarder et charger les statistiques de normalisation avec le checkpoint.

---

### 1.4 Curriculum Learning — Fait, Insuffisant (v3_curriculum)

**État actuel (avant v3_curriculum) :** La tâche est présentée à pleine difficulté
dès le premier épisode. `theta_success=0.90` n'est jamais atteint.

**Amélioration implémentée :** `theta_success=0.40` + spawn déviation réduite
(`pos_x: 0.05`, `pos_y: 0.02`) + pénalités adoucies (`w_wrong_dir: 0.05`,
`w_stagnation: 0.02`).

**Résultat observé :**
- 300k steps : `val_return_mean` = +12.6, `val_door_angle_final_mean` = 0.021 rad — signal positif
- 500k steps : retour à 0.017 rad — régression après le pic
- `success_frac = 0` tout au long — même 0.40 jamais atteint
- `theta_best_mean` training = 0.002–0.004 rad — porte quasi-immobile même avec exploration

**Verdict :** Le curriculum a apporté un signal positif transitoire mais n'a pas
résolu le problème fondamental — le replay buffer reste vide de succès confirmés.
Le critic ne valorise pas l'ouverture de porte car il n'a jamais vu de transition
avec reward de succès.

**Lien cours :** Curriculum learning — efficace pour guider l'exploration, mais
insuffisant quand la reward sparse elle-même n'est jamais déclenchée. Nécessite
d'être combiné avec HER pour ce type de tâche.

---

### 1.4b HER (Hindsight Experience Replay) — En cours

**Motivation :** Après v1→v2→v3→curriculum, le diagnostic est clair : le replay
buffer ne contient jamais de transition avec reward de succès (`success_frac=0`
sur 2.3M steps cumulés). Sans signal positif, le critic ne peut pas apprendre à
valoriser l'ouverture de la porte.

**Amélioration :** `HerReplayBuffer` avec `GoalConditionedWrapper` :
- Observation augmentée : `{observation, achieved_goal=[θ], desired_goal=[θ_success]}`
- Reward sparse : `0` si `θ ≥ θ_success`, `−1` sinon
- Pour chaque transition réelle, 4 transitions virtuelles avec `desired_goal` = angles futurs atteints dans l'épisode
- Si la porte a atteint 0.01 rad pendant l'épisode, des succès virtuels sont créés pour `desired_goal=0.01 rad`

**Config :** `configs/train/open_single_door_sac_her.yaml`
- `theta_success: 0.15 rad` (objectif HER, plus accessible que 0.40)
- `her_n_sampled_goal: 4`, `her_goal_strategy: future`
- `MultiInputPolicy` (gère les observations dict)

**Implémentation :**
- `GoalConditionedWrapper` dans `envs/factory.py` (interface GoalEnv SB3)
- `use_her: true` dans la config train → `HerReplayBuffer` automatique dans `_build_sac`

**Lien cours :** Off-policy learning avec replay buffer de qualité. HER (Andrychowicz
et al., 2017) est la solution standard pour les tâches de manipulation avec reward
sparse — permet d'apprendre même quand aucun succès réel n'est observé.

---

### 1.5 Behavioral Cloning Pre-training + SAC Fine-tuning

**État actuel :** La politique démarre avec des poids aléatoires. Le premier
contact utile avec la poignée est découvert par exploration aléatoire — processus
très inefficace sur un espace d'action 12D et une cible de 5 cm.

**Amélioration :**

1. **Collecte de démonstrations** via les politiques scriptées RoboCasa ou
   télé-opération (`robocasa/demos/`).
2. **Pre-training BC** — entraîner le réseau acteur sur les paires `(état, action)`
   avec une loss MSE supervisée.
3. **Fine-tuning SAC** — initialiser SAC avec les poids BC ; le RL corrige le
   distribution shift et dépasse les performances de BC seul.

```python
# Schéma conceptuel
demos = collect_demos("OpenCabinet", n=50)
bc_train(actor_network, demos, epochs=100)
model = SAC.load_pretrained(actor_network, ...)
model.learn(total_timesteps=3_000_000)
```

**Lien cours :** Imitation learning — Behavioral Cloning, covariate shift, et
l'intérêt de combiner pre-training par démonstration avec fine-tuning RL pour
éviter l'exploration aléatoire pure.

**Impact attendu :** Succès non nul dès la première évaluation (~step 25k).
Réduction drastique du compute gaspillé en exploration aléatoire.

---

### 1.6 Ablations sur les Composantes Reward

**État actuel :** Le reward shaping a 7 composantes. L'impact individuel de
chaque composante est inconnu.

**Amélioration :** Ablation systématique — supprimer une composante à la fois
et mesurer l'effet sur `approach_frac_mean`, `val_success_rate`, et
`stagnation_steps_mean` :

| Ablation | Config | Hypothèse |
|---|---|---|
| Sans pénalité stagnation (`w_stagnation=0`) | `reward.w_stagnation: 0.0` | `stagnation_steps_mean` augmente, `success_rate` baisse |
| Sans gating approche (toujours actif) | Modifier `reward.py` | `approach_frac` augmente, hover hacking réapparaît |
| Sans high-watermark (`r_progress = θ_t`) | Modifier `reward.py` | `sign_changes_mean` augmente, oscillation profitable |
| Sans pénalité oscillation (`w_oscillation=0`) | `reward.w_oscillation: 0.0` | `sign_changes_mean` augmente |
| `gradient_steps=1` vs `gradient_steps=4` vs `gradient_steps=12` | Modifier config | Trade-off sample efficiency vs wall-clock |
| `n_envs=1` vs `n_envs=6` vs `n_envs=12` | Modifier config | Impact du parallélisme sur la qualité de l'entraînement |

**Lien cours :** Les ablations sont l'outil standard pour comprendre quelles
composantes d'une méthode contribuent à la performance. Le cours les couvre dans
le contexte du reward design et de l'analyse d'algorithmes.

---

### 1.7 Multi-Seed Evaluation

**État actuel :** Toutes les expériences utilisent un seed unique (seed=0). Les
résultats RL sont fortement dépendants du seed.

**Amélioration :**

```bash
for seed in 0 1 2 3 4; do
    make train-sac-v3 SEED=$seed
done
```

Reporter mean ± std sur les 5 seeds pour `val_success_rate`, `door_angle_final_mean`,
et `best_validation_step`.

**Lien cours :** Robustesse statistique en RL — le cours souligne que les résultats
sur un seul seed sont des anecdotes, pas des preuves. Le standard de la littérature
est 5–10 seeds avec intervalles de confiance.

---

### 1.8 Comparaison TD3

**État actuel :** SAC (principal) + PPO (baseline) implémentés. Pas de TD3.

**Amélioration :** Ajouter TD3 comme troisième algorithme :

```yaml
# configs/train/open_single_door_td3.yaml
train:
  algorithm: TD3
  learning_rate: 3.0e-4
  buffer_size: 1_000_000
  batch_size: 256
  tau: 0.005
  policy_delay: 2
  target_policy_noise: 0.2
  noise_clip: 0.5
```

TD3 est off-policy actor-critic comme SAC, mais sans régularisation entropique.
Comparer SAC et TD3 isole la contribution de l'entropie à l'exploration.

**Lien cours :** Off-policy actor-critic — comprendre les trade-offs entre SAC
(politique stochastique + entropie), TD3 (politique déterministe + bruit d'exploration),
et PPO (on-policy).

---

### 1.9 Replay Buffer Inspection

**État actuel :** Le replay buffer SAC est sauvegardé en `.pkl` mais jamais analysé.

**Amélioration :** Analyser périodiquement la composition du buffer :
- Distribution des rewards (la majorité des transitions est-elle à reward ≈ 0 ?)
- Distribution des `door_angle` dans le buffer (y a-t-il des transitions à θ élevé ?)
- Taux de succès parmi les transitions bufferisées

Si le buffer ne contient aucune transition avec θ > 0.5, l'agent n'a jamais ouvert
la porte assez loin pour apprendre le comportement final.

**Lien cours :** Off-policy learning — la qualité du replay buffer détermine
directement ce que l'agent peut apprendre. Un buffer composé à 99% de transitions
à reward ≈ 0 indique une exploration inefficace.

---

### 1.10 Critère de Succès Soutenu

**État actuel :** Succès = `θ ≥ 0.90` à n'importe quel step de l'épisode. Un
agent peut atteindre momentanément le seuil puis relâcher la porte.

**Améliorations possibles :**
- **Succès soutenu** : requérir `θ ≥ 0.90` pendant ≥ 10 steps consécutifs.
- **Succès en état final** : requérir `θ ≥ 0.90` au dernier step de l'épisode.
- **Succès hors distribution** : évaluer sur des types de placards et positions
  non vus pendant l'entraînement.

**Lien cours :** Protocole d'évaluation — la définition du succès affecte toutes
les métriques reportées. Le cours couvre l'importance d'aligner le critère
d'évaluation avec l'objectif réel de la tâche.

---

### 1.11 Policy Evaluation on Held-Out Configurations

**État actuel :** Le split test utilise `seed=20000` qui randomise l'apparence de
la scène, mais pas le type de placard ni la plage de positions initiales.

**Amélioration :** Définir des ensembles d'évaluation réellement hors distribution :
- Styles de placard différents (non vus pendant l'entraînement)
- Distances robot-placard différentes
- Conditions d'éclairage différentes

**Lien cours :** Généralisation en robot learning — une politique qui fonctionne
uniquement pour la distribution d'entraînement n'est pas utile en déploiement.

---

## 2. Roadmap Prioritaire

### Fait (avant deadline)

- `ent_coef=0.1` fixe (v3) — résout le crash d'auto-tuning de v1/v2
- `use_sde=True`, `sde_sample_freq=64` (v3) — exploration cohérente
- `gradient_steps=4`, `tau=0.01` (v2/v3) — stabilité du critic
- LR cosine schedule 3e-4 → 1e-5 (v3)
- Curriculum learning : `theta_success=0.40`, spawn déviation réduite (v3_curriculum)
- HER : `GoalConditionedWrapper` + `HerReplayBuffer`, `theta_success=0.15` (en cours)

### Post-deadline prioritaire

- Observation normalisation (`VecNormalize`) — faible effort, fort impact
- Multi-seed (3–5 seeds) — résultats statistiquement valides
- Ablations reward — quantifier l'utilité de chaque composante

### Long terme

- BC pre-training + SAC fine-tuning — résout définitivement le cold start
- Évaluation hors distribution — test de généralisation réelle
- Intégration navigation + base mobile — tâche complète PandaOmron
