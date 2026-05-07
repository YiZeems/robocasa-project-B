# Courbes d'entraînement — RoboCasa OpenCabinet SAC

Métriques exportées depuis MLflow. Run principal : `OpenCabinet_SAC_seed0_20260507_073628`, algorithme SAC, tâche `OpenCabinet`.

---

## Train — Métriques internes SAC (6)

Ces métriques sont loguées à chaque gradient update (~65k points).

### train_01 — Actor Loss
![Actor Loss](run_sac_3M/train_01_actor_loss.png)
Perte de l'acteur SAC (policy gradient). Doit être négative et décroître.

### train_02 — Critic Loss
![Critic Loss](run_sac_3M/train_02_critic_loss.png)
Erreur TD du critic (MSE sur la valeur Q). Doit converger vers 0.

### train_03 — Entropy Coefficient (α)
![Entropy Coef](run_sac_3M/train_03_ent_coef.png)
Coefficient d'entropie automatique — contrôle l'exploration. Diminue quand la politique devient plus déterministe.

### train_04 — Entropy Coef Loss
![Entropy Coef Loss](run_sac_3M/train_04_ent_coef_loss.png)
Gradient de l'ajustement automatique de α.

### train_05 — Learning Rate
![Learning Rate](run_sac_3M/train_05_learning_rate.png)
Taux d'apprentissage (constant ici à 3e-4).

### train_06 — Gradient Updates (n_updates)
![N Updates](run_sac_3M/train_06_n_updates.png)
Nombre cumulé de mises à jour du gradient.

---

## Reward Hack Monitor — Métriques par épisode d'entraînement (12)

Ces métriques sont loguées à chaque `eval_freq` steps depuis les épisodes d'entraînement rolling (7 points = 7 évaluations).

### rh_01 — Train Success Rate
![Train Success Rate](run_sac_3M/rh_01_train_success_rate.png)
Taux de succès sur les épisodes d'entraînement (rolling). Indicateur précoce avant la validation.

### rh_02 — Success Fraction
![Success Frac](run_sac_3M/rh_02_success_frac.png)
Fraction d'épisodes terminés avec succès (porte ouverte à ≥ 90%).

### rh_03 — Best Door Angle (mean)
![Theta Best](run_sac_3M/rh_03_theta_best_mean.png)
Meilleur angle de porte atteint en moyenne par épisode. Normalisé 0→1, succès à 0.90. Indicateur de progression même sans succès complet.

### rh_04 — Approach Fraction
![Approach Frac](run_sac_3M/rh_04_approach_frac.png)
Fraction de la récompense provenant du guidage vers la poignée. Doit rester faible — une valeur élevée indique du **hover-hacking** (l'agent tourne autour sans ouvrir).

### rh_05 — Progress Fraction
![Progress Frac](run_sac_3M/rh_05_progress_frac.png)
Fraction de la récompense provenant de l'ouverture progressive (signal principal). Doit dominer en milieu d'entraînement.

### rh_06 — Reward Without Success Bonus
![Reward w/o Success](run_sac_3M/rh_06_reward_without_success.png)
Récompense shapée sans le bonus sparse de succès. Utile pour détecter si l'agent exploite le shaping sans vraiment réussir.

### rh_07 — Oscillation Fraction
![Oscillation Frac](run_sac_3M/rh_07_oscillation_frac.png)
Fraction d'épisodes présentant des oscillations détectées. Doit être faible.

### rh_08 — Oscillation Steps (mean)
![Oscillation Steps](run_sac_3M/rh_08_oscillation_steps_mean.png)
Nombre moyen de steps en oscillation par épisode.

### rh_09 — Sign Changes (mean)
![Sign Changes](run_sac_3M/rh_09_sign_changes_mean.png)
Nombre moyen de changements de direction de la porte par épisode. Indicateur d'oscillation de la politique.

### rh_10 — Stagnation Rate
![Stagnation Rate](run_sac_3M/rh_10_stagnation_rate.png)
Taux d'épisodes stagnants (porte qui ne bouge plus). Doit être faible en fin d'entraînement.

### rh_11 — Stagnation Steps (mean)
![Stagnation Steps](run_sac_3M/rh_11_stagnation_steps_mean.png)
Nombre moyen de steps de stagnation par épisode.

### rh_12 — Episodes Logged
![Episodes](run_sac_3M/rh_12_episodes.png)
Nombre d'épisodes accumulés dans la fenêtre rolling de monitoring.

---

## Validation — Métriques de validation (19 courbes)

Évaluées sur des épisodes avec seed fixe (`validation_seed=10000`) toutes les `eval_freq` steps.

### val_01 — Success Rate
![Success Rate](run_sac_3M/val_01_success_rate.png)
Taux de succès sur les épisodes de validation. Métrique principale.

### val_02 — Success Rate + Intervalle de Confiance
![Success Rate CI](run_sac_3M/val_02_success_rate_ci.png)
Success rate avec bornes basse et haute de l'intervalle de confiance.

### val_03 — Failure Rate
![Failure Rate](run_sac_3M/val_03_failure_rate.png)
Taux d'échec (1 - success rate). Complémentaire pour voir les régessions.

### val_04 — Episode Return (mean)
![Return Mean](run_sac_3M/val_04_return_mean.png)
Récompense cumulée moyenne par épisode de validation.

### val_05 — Episode Return (mean / median / std)
![Return Full](run_sac_3M/val_05_return_full.png)
Vue complète du retour : moyenne, médiane, écart-type.

### val_06 — Door Angle Final (mean)
![Door Angle Final](run_sac_3M/val_06_door_angle_final_mean.png)
Angle final de la porte en fin d'épisode (normalisé). Succès à ≥ 0.90.

### val_07 — Door Angle (final mean / final std / max mean)
![Door Angle Full](run_sac_3M/val_07_door_angle_full.png)
Vue complète : angle final moyen, dispersion, et meilleur angle atteint.

### val_08 — Episode Length (mean)
![Episode Length](run_sac_3M/val_08_episode_length_mean.png)
Longueur moyenne des épisodes. Diminue quand l'agent réussit plus vite.

### val_09 — Episode Length (mean / std)
![Episode Length Full](run_sac_3M/val_09_episode_length_full.png)
Longueur avec dispersion.

### val_10 — Action Magnitude (mean)
![Action Magnitude](run_sac_3M/val_10_action_magnitude_mean.png)
Norme L2 moyenne des actions — indique si la politique est agressive ou douce.

### val_11 — Action Magnitude (mean / std)
![Action Magnitude Full](run_sac_3M/val_11_action_magnitude_full.png)
Magnitude avec dispersion.

### val_12 — Action Smoothness
![Action Smoothness](run_sac_3M/val_12_action_smoothness.png)
Score de fluidité des actions (moins de jerks = meilleur contrôle).

### val_13 — Approach Fraction
![Approach Frac](run_sac_3M/val_13_approach_frac.png)
Fraction de récompense d'approche sur les épisodes de validation. Doit rester faible.

### val_14 — Reward Without Success
![Reward w/o Success](run_sac_3M/val_14_reward_without_success.png)
Récompense shapée sans bonus de succès (validation).

### val_15 — Sign Changes (mean)
![Sign Changes](run_sac_3M/val_15_sign_changes.png)
Changements de direction de la porte sur les épisodes de validation.

### val_16 — Stagnation Steps (mean)
![Stagnation Steps](run_sac_3M/val_16_stagnation_steps.png)
Steps de stagnation par épisode de validation.

### val_17 — Num Episodes Run
![Num Episodes](run_sac_3M/val_17_num_episodes.png)
Nombre d'épisodes évalués à chaque checkpoint.

### val_18 — Success CI Low
![CI Low](run_sac_3M/val_18_success_ci_lo.png)
Borne basse de l'intervalle de confiance du success rate.

### val_19 — Success CI High
![CI High](run_sac_3M/val_19_success_ci_hi.png)
Borne haute de l'intervalle de confiance du success rate.
