# Model Improvements and Future Work

> This document identifies concrete improvements to the current project, each linked to a concept from the IA705 — Learning for Robotics course.

---

## 1. Improvements Directly Linked to Course Concepts

### Summary Table

| Amélioration | Motivation | Impact attendu | Difficulté | Priorité | Concept cours |
|---|---|---|---|---|---|
| Fix `ent_coef` : valeur fixe (v3) | Auto-tuning crash → α→0 → politique déterministe | Apprentissage stable, exploration maintenue | Faible | **Critique (fait)** | Exploration vs exploitation |
| `use_sde=True` (v3) | Bruit corrélé à l'état → mouvements cohérents vers la poignée | Premier contact plus rapide | Faible | **Haute (fait)** | Exploration structurée |
| `gradient_steps=4` + `tau=0.01` (v2/v3) | Critic loss spikes à 500k steps → divergence | Stabilité de l'entraînement | Faible | **Critique (fait)** | Actor-critic — stabilité du critic |
| `eval_freq=100k` + `n_eval_episodes=10` | Validation = 82% du wall time en v1 | Throughput ×3–4 | Faible | **Haute (fait)** | Protocole expérimental |
| `obj_registries=[lightwheel]` | Reset 9.3s vs 4.8s avec assets légers | Throughput ×1.9 | Faible | **Haute (fait)** | Efficacité computationnelle |
| `control_freq=20` | Simulation 2× plus lente que nécessaire | Throughput ×2 | Faible | **Haute (fait)** | Simulation physique |
| Observation normalisation (`VecNormalize`) | Observations brutes sur 220D de scales très différentes | Convergence SAC plus rapide, gradients stables | Faible | Haute | Variance reduction — policy gradient |
| Curriculum learning (initial door angle) | Tâche pleine difficulté dès le départ → exploration peu efficace | Premier succès < 100k steps | Moyenne | Haute | Curriculum learning |
| BC pre-training + SAC fine-tuning | Exploration aléatoire gaspille des milliers d'épisodes | Succès non nul dès les premiers épisodes | Moyenne | Haute | Imitation learning |
| Ablations sur les composantes reward | Impact individuel de chaque composante inconnu | Simplification du reward + compréhension | Faible–Moyenne | Haute | Reward shaping — analyse |
| Multi-seed (3–5 seeds) | Seed unique → variance inconnue | Conclusions statistiquement valides | Faible (coût compute) | Haute | Robustesse statistique |
| Comparaison TD3 | Isole la contribution de l'entropie (SAC vs TD3) | Compréhension de l'exploration entropique | Faible | Moyenne | Off-policy actor-critic |
| Replay buffer inspection | Buffer peut contenir 100% de transitions sans signal | Diagnostic de la qualité du buffer | Faible | Moyenne | Off-policy learning |
| Critère de succès soutenu | Succès momentané ≠ porte maintenue ouverte | Alignement évaluation / objectif réel | Faible | Moyenne | Protocole d'évaluation |
| Évaluation sur configs hors distribution | Tâche validée sur les seeds d'entraînement uniquement | Test de généralisation réelle | Moyenne | Moyenne | Généralisation en robotique |
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

### 1.4 Curriculum Learning

**État actuel :** La tâche est présentée à pleine difficulté dès le premier épisode —
porte entièrement fermée, robot en position initiale. Les premiers contacts utiles
avec la poignée sont découverts par chance.

**Amélioration :** Curriculum en 3 étapes sur l'angle initial de la porte :

```python
# Étape 1 : porte pré-ouverte à 50%
initial_theta = 0.50

# Étape 2 : porte pré-ouverte à 20%
initial_theta = 0.20

# Étape 3 : tâche complète, porte fermée
initial_theta = 0.00
```

Implémentation dans `envs/factory.py` en passant `initial_door_angle` à
l'initialisation de l'environnement RoboCasa.

**Lien cours :** Curriculum learning — la difficulté progressive réduit le défi
d'exploration. L'agent accumule la reward de succès plus tôt, rendant le credit
assignment tractable dès le début de l'entraînement.

**Impact attendu :** Premier succès en < 100k steps (vs plusieurs centaines de
milliers sans curriculum). Taux de succès final plus élevé.

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

### Court terme (avant ou juste après la deadline)

1. Observation normalisation (`VecNormalize`) — faible effort, impact élevé sur la
   stabilité de SAC v3.
2. Analyse du replay buffer en cours de run — diagnostic sur la qualité du buffer
   v3.

### Moyen terme (1–4 semaines post-deadline)

3. Multi-seed (3–5 seeds) — transforme les résultats anecdotiques en résultats
   statistiquement valides.
4. Ablations sur les composantes reward — quantifie l'utilité de chaque composante.
5. Curriculum learning — amélioration la plus impactante sur le taux de convergence.

### Long terme (1 semestre post-deadline)

6. BC pre-training + SAC fine-tuning — transformerait ce projet de cours en
   contribution publiable sur l'efficacité d'exploration.
7. Évaluation sur configurations hors distribution — test de généralisation réelle.
8. Intégration navigation + base mobile — tâche complète PandaOmron.
