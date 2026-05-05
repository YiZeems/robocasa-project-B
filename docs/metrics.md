# Métriques — Référence complète

> Toutes les métriques sont loggées dans MLflow (`mlruns/`).  
> Les métriques de validation sont aussi dans `outputs/<run_id>/validation_curve.csv`.

---

## 1. Métriques de performance

### `val_success_rate` / `success_rate`

**Définition :** proportion d'épisodes dans lesquels l'agent atteint θ ≥ 0.90 au moins une fois.

**Pourquoi :** métrique principale du projet. C'est elle qui détermine quel checkpoint est sauvegardé comme `best_model.zip`.

**Interpréter :**
- 0 % → l'agent n'ouvre jamais la porte (hover hacking probable)
- 30–50 % → convergence partielle, encore beaucoup de variabilité
- > 80 % → objectif validé sur le split validation

**Erreurs révélées :** si `success_rate ≈ 0` mais `return_mean` est élevé → reward hacking.

**Dans MLflow :** `validation/val_success_rate` (périodique), `eval_video/success_rate` (post-training)

**Intervalle de confiance :** Wilson score 95 % calculé dans `metrics.py::success_confidence_interval()`. Colonnes : `val_success_ci_lo`, `val_success_ci_hi`.

---

### `val_return_mean` / `return_mean`

**Définition :** return cumulé moyen par épisode sur N épisodes de validation.

**Pourquoi :** indicateur de progrès global. Devrait croître au fil du training et se stabiliser.

**Interpréter :**
- Croissance monotone → bonne convergence
- Plateau soudain → early stopping probable
- Croissance sans `success_rate` → reward hacking

**Erreurs révélées :** return élevé + succès nul = indicateur quasi-certain de reward gaming.

**Dans MLflow :** `validation/val_return_mean`

---

### `val_episode_length_mean`

**Définition :** longueur moyenne des épisodes (en steps) sur validation.

**Pourquoi :** une longueur décroissante indique que l'agent réussit plus vite (ou échoue plus vite). Une longueur constante à 500 indique que l'agent épuise l'horizon sans succès.

**Interpréter :**
- Décroissante → agent plus efficace
- Constante à 500 → agent échoue systématiquement

**Dans MLflow :** `validation/val_episode_length_mean`

---

### `val_door_angle_final_mean`

**Définition :** angle de porte normalisé [0, 1] à la **fin** de l'épisode, moyenné sur N épisodes.

**Pourquoi :** mesure l'ouverture atteinte en moyenne à la fin de l'épisode. Plus fiable que `success_rate` seul pour distinguer "0 % d'ouverture" de "80 % d'ouverture sans succès".

**Interpréter :**
- < 0.2 → agent bloqué loin de la solution
- 0.5–0.8 → progression mais pas de succès consistant
- > 0.9 → succès ou très proche

**Erreurs révélées :** si `door_angle_final_mean` ≈ 0.05 après 1M steps → hover hacking ou mauvaise initialisation.

**Dans MLflow :** `validation/val_door_angle_final_mean`

---

### `val_door_angle_max_mean`

**Définition :** angle de porte maximal atteint **à un moment quelconque** de l'épisode (high-watermark), moyenné sur N épisodes.

**Pourquoi :** révèle si l'agent approche de la solution sans maintenir l'ouverture. Si `door_angle_max_mean` est élevé mais `door_angle_final_mean` est faible → l'agent ouvre puis referme.

**Interpréter :**
- `max >> final` → oscillation / refermeture active
- `max ≈ final` → l'agent maintient l'ouverture (bon signe)
- `max < 0.3` → agent bloqué

**Dans MLflow :** `validation/val_door_angle_max_mean`

---

### `val_door_angle_delta`

**Définition :** `door_angle_final − door_angle_initial` par épisode.

**Pourquoi :** mesure la progression nette. Élimine les biais dus à une initialisation à un angle déjà partiellement ouvert.

**Dans MLflow :** `eval_video/door_angle_delta` (dans le debug CSV)

---

### `time_to_success`

**Définition :** step auquel l'agent atteint θ ≥ 0.90 pour la première fois dans l'épisode. `-1` si pas de succès.

**Pourquoi :** mesure l'efficacité temporelle. Un agent qui réussit au step 50 est meilleur qu'un agent qui réussit au step 490 (même horizon de 500).

**Dans MLflow :** `eval_video/time_to_success` (dans le debug CSV)

---

## 2. Métriques anti-hacking

### `val_approach_frac_mean`

**Définition :** `approach_total / max(reward_total, ε)` — fraction de la reward totale due à la composante d'approche.

**Pourquoi :** indicateur direct de hover hacking. Si l'agent gagne principalement de la reward en restant proche de la poignée plutôt qu'en ouvrant la porte, cette fraction est élevée.

**Interpréter :**
- < 0.2 → sain (reward dominée par progress + success)
- 0.3–0.5 → à surveiller
- > 0.5 → alerte hover hacking

**Dans MLflow :** `validation/val_approach_frac_mean`

---

### `val_stagnation_steps_mean`

**Définition :** nombre moyen de steps par épisode où le robot est près de la poignée (d < 0.12 m) sans progresser (ctr ≥ 50).

**Pourquoi :** mesure la tendance à stagner devant la poignée sans agir.

**Interpréter :**
- < 10 → sain
- 50–100 → l'agent passe du temps à stagner
- > 100 → hover hacking chronique

**Dans MLflow :** `validation/val_stagnation_steps_mean`

---

### `val_sign_changes_mean`

**Définition :** nombre moyen de changements de signe de Δθ par épisode (fenêtre glissante de 20 steps).

**Pourquoi :** mesure l'oscillation de la porte. Un robot qui ouvre et referme alternativement produit beaucoup de changements de signe.

**Interpréter :**
- < 5 → sain
- 5–15 → oscillation modérée
- > 15 → oscillation pathologique

**Dans MLflow :** `validation/val_sign_changes_mean`

---

### `reward_without_success`

**Définition :** return moyen des épisodes en échec (success=False).

**Pourquoi :** si cette valeur est élevée (proche du return global), l'agent accumule de la reward même sans accomplir la tâche — indicateur de reward gaming.

**Interpréter :**
- Proche de 0 → les épisodes ratés rapportent peu (bon signe)
- Élevé + `success_rate` faible → gaming sévère

**Dans MLflow :** `validation/val_reward_without_success`

---

## 3. Métriques de qualité des actions

### `val_action_smoothness_mean`

**Définition :** `mean(‖a_t − a_{t-1}‖)` sur l'épisode. Mesure la jerkiness (saccades).

**Pourquoi :** une politique apprise avec régularisation action (`w_action_reg > 0`) devrait produire des actions lisses. Des actions saccadées peuvent indiquer une policy instable ou une reward mal balancée.

**Interpréter :**
- Faible et stable → politique fluide
- Croissant → la politique devient plus agressive (potentiellement mauvais signe)

**Dans MLflow :** `validation/val_action_smoothness_mean`

---

### `val_action_magnitude_mean`

**Définition :** `mean(‖a_t‖)` par épisode.

**Pourquoi :** complète `action_smoothness`. Des actions de grande magnitude peuvent indiquer que l'agent "pousse fort" sans finesse.

**Dans MLflow :** `validation/val_action_magnitude_mean`

---

## 4. Métriques d'algorithme

### SAC

| Métrique | Nom MLflow | Interprétation |
|---|---|---|
| Perte acteur | `train/actor_loss` | Devrait être négatif et décroissant |
| Perte critique (Q1+Q2) | `train/critic_loss` | Devrait converger |
| Perte entropie | `train/entropy_loss` | Suit l'évolution du coefficient ent_coef |
| Valeur Q moyenne | `train/q_values` | Doit croître avec l'apprentissage |
| Coefficient entropie | `train/ent_coef` | Doit converger vers une valeur stable |

### PPO

| Métrique | Nom MLflow | Interprétation |
|---|---|---|
| KL approx. | `train/approx_kl` | Doit rester < 0.05 (si > 0.1 → step trop grand) |
| Clip fraction | `train/clip_fraction` | Doit rester < 0.3 (si > 0.5 → clip_range trop petit) |
| Perte valeur | `train/value_loss` | Décroissante |
| Perte politique | `train/policy_gradient_loss` | Négatif |
| Perte entropie | `train/entropy_loss` | Légèrement négatif |

---

## 5. Métriques computationnelles

### `train/fps`

**Définition :** steps d'environnement par seconde.

**Valeurs typiques :**
- SAC, 12 workers, MPS (Apple Silicon) : ~2 600 steps/min
- SAC, 12 workers, RTX 4070 : ~3 500 steps/min
- PPO, 1 worker, CPU : ~800 steps/min

**Interprétation :** une chute soudaine de FPS peut indiquer un goulot d'étranglement CPU (trop de gradient steps) ou un problème mémoire.

---

## 6. Métriques de reproductibilité

| Métrique | Où la trouver | Usage |
|---|---|---|
| `seed` | `train_summary.json` | Permet de reproduire exactement |
| `best_validation_step` | `train_summary.json` | Step du best_model.zip |
| `best_validation_success` | `train_summary.json` | Success rate du best checkpoint |
| `total_timesteps_done` | `train_summary.json` | Nombre de steps réellement effectués |
| `run_id` | Convention `task_algo_seed_timestamp` | Identifiant unique du run |

---

## 7. Localisation dans le code

| Métrique | Fichier source | Ligne/fonction |
|---|---|---|
| `summarize_rollout_episodes` | `utils/metrics.py` | Calcule toutes les métriques de rollout |
| `success_confidence_interval` | `utils/metrics.py` | Wilson 95% CI |
| `ValidationCallback._log_metrics` | `rl/train.py` | Log vers MLflow + CSV |
| `RewardHackMonitorCallback` | `rl/train.py` | Métriques intra-épisode anti-hacking |
| `AntiHackingReward.episode_summary` | `envs/reward.py` | Compteurs par épisode |

---

## 8. `validation_curve.csv` — colonnes

Toutes les métriques de validation sont exportées dans `outputs/<run_id>/validation_curve.csv` :

```
step, val_success_rate, val_success_ci_lo, val_success_ci_hi,
val_return_mean, val_return_std, val_return_median,
val_episode_length_mean, val_episode_length_std,
val_door_angle_final_mean, val_door_angle_final_std,
val_door_angle_max_mean, val_sign_changes_mean,
val_stagnation_steps_mean, val_approach_frac_mean,
val_action_magnitude_mean, val_action_smoothness_mean,
val_reward_without_success
```

Utilisé par `scripts/plot_training.py` pour générer les courbes PNG.
