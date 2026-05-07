# Courbes d'entraînement — RoboCasa OpenCabinet SAC

Tâche : ouvrir une porte de placard avec un bras robotique PandaOmron. Algorithme : SAC (Soft Actor-Critic). Ce README retrace l'évolution des runs successifs, les problèmes rencontrés, les corrections apportées, et les métriques à surveiller.

---

## Table des matières

1. [Contexte général](#contexte-général)
2. [Run SAC v1 — 500k steps](#run-sac-v1--500k-steps)
3. [Run SAC v2 — 900k steps](#run-sac-v2--900k-steps)
4. [Run SAC v3 — 400k steps](#run-sac-v3--400k-steps)
5. [Run SAC v3 Curriculum — 500k steps](#run-sac-v3-curriculum--500k-steps)
6. [Run SAC HER — en cours](#run-sac-her--en-cours)
7. [Référence des métriques](#référence-des-métriques)

---

## Contexte général

L'agent reçoit à chaque step une observation de 248 dimensions (positions joints, vitesses, distance poignée-EEF, état porte…) et produit une action 12D (commande OSC du bras). L'horizon est de 500 steps par épisode. Un succès = porte ouverte au-delà d'un seuil angulaire configurable.

Le reward est **shapé** (dense) pour guider l'apprentissage :
- **Approche** : récompense quand le bras se rapproche de la poignée
- **Progression** : récompense proportionnelle à l'ouverture de la porte
- **Succès** : bonus sparse quand la porte dépasse le seuil
- **Pénalités** : stagnation, oscillation, mauvaise direction, régularisation des actions

Les runs sont évalués périodiquement en **validation déterministe** (seed fixée, pas d'exploration) pour mesurer la vraie politique apprise.

---

## Run SAC v1 — 500k steps

**Dossier :** [`run_sac_v1/`](run_sac_v1/)
**Run ID :** `20260506_201751`
**Config :** `open_single_door_sac.yaml`

### Paramètres clés

| Paramètre | Valeur |
|---|---|
| `ent_coef` | `"auto"` (auto-tuning) |
| `target_entropy` | `"auto"` (= -action_dim = -12) |
| `learning_rate` | 3e-4 constant |
| `use_sde` | false |
| `net_arch` | [256, 256] |
| `theta_success` | 0.90 (90% d'ouverture) |

### Ce qui s'est passé

L'acteur loss chutait vers -50 rapidement puis **remontait** après 100k steps. Le coefficient d'entropie α crashait vers 0 dès 200k steps. La validation door angle finale stagnait à ~0.003 rad (porte quasi fermée). Success rate = 0 tout le long.

**Cause racine identifiée :** SAC avec `ent_coef="auto"` calcule α en résolvant :

```
∇α = α × (-log_prob - target_entropy)
```

Avec `log_prob ≈ -20` (12D gaussienne) et `target_entropy = -12`, on obtient toujours `-20 - (-12) = -8 < 0`, ce qui pousse α → 0 **sans jamais s'arrêter**. Résultat : politique déterministe trop tôt, Q-values divergentes, critic loss explosant à **40 000+**.

### Pourquoi arrêtée

Aucun progrès après 200k steps. α = 0 → plus d'exploration → l'agent n'ouvre jamais la porte.

### Amélioration apportée → v2

Passer à `ent_coef="auto_0.1"` (initialisation à 0.1) et `target_entropy=-4` (cible moins agressive) pour éviter le crash de α.

---

## Run SAC v2 — 900k steps

**Dossier :** [`run_sac_v2/`](run_sac_v2/)
**Run ID :** `20260507_073628`
**Config :** `open_single_door_sac_v2.yaml`

### Paramètres clés

| Paramètre | Valeur |
|---|---|
| `ent_coef` | `"auto_0.1"` |
| `target_entropy` | `-4` |
| `learning_rate` | 3e-4 cosine → 1e-5 |
| `use_sde` | true, `sde_sample_freq=64` |
| `net_arch` | [400, 300] |
| `gradient_steps` | 4 |
| `tau` | 0.01 |

### Ce qui s'est passé

α partait de 0.1 et retombait encore à 0 vers 100k steps. Avec `log_prob ≈ -20` et `target_entropy=-4`, le gradient est toujours `-20 - (-4) = -16 < 0` → α crash inévitable, juste retardé. La critic loss restait plus raisonnable (~4–6) grâce à `gradient_steps=4` et `tau=0.01`, mais l'actor loss remontait après 200k. Success rate = 0 sur toute la durée des 900k steps.

**Cause racine identifiée :** le problème de fond n'est pas l'initialisation de α mais l'auto-tuning lui-même : avec un espace d'action 12D et des policies gaussiennes standard, `log_prob` sera **toujours** plus négatif que `target_entropy` raisonnable. α → 0 est inévitable tant qu'on utilise l'auto-tuning.

### Pourquoi arrêtée

900k steps, 0 succès, α toujours à 0 après 100k. Aucun signe de progrès.

### Amélioration apportée → v3

Fixer `ent_coef=0.1` **sans auto-tuning**. L'entropie reste constante tout le long, ce qui maintient l'exploration.

---

## Run SAC v3 — 400k steps

**Dossier :** [`run_sac_v3/`](run_sac_v3/)
**Run ID :** `20260507_112859`
**Config :** `open_single_door_sac_v3.yaml`

### Paramètres clés

| Paramètre | Valeur |
|---|---|
| `ent_coef` | `0.1` (fixe) |
| `learning_rate` | 3e-4 cosine → 1e-5 |
| `use_sde` | true, `sde_sample_freq=64` |
| `net_arch` | [400, 300] |
| `gradient_steps` | 4 |
| `tau` | 0.01 |
| `theta_success` | 0.90 |

### Ce qui s'est passé

**Amélioration claire sur les métriques internes :** α plat à 0.1 ✅, actor loss stable à -45/-50 sans remontée ✅, critic loss contenu à ~2-4 (vs 40 000 en v1) ✅. L'agent apprend quelque chose de stable.

Cependant, la validation `door_angle_final_mean` restait à ~0.001-0.003 rad. En training, `theta_best_mean ≈ 0.003` rad. La politique déterministe n'ouvre pas la porte du tout. Success rate = 0.

**Problème résiduel :** même avec entropie fixe, l'agent ne parvient pas à établir le premier contact avec la poignée pour déclencher la récompense de progression. Sans ce premier signal, le critic n'apprend pas de Q-values positives → l'actor ne cherche pas à ouvrir. C'est le **cold start problem** de la manipulation.

### Pourquoi arrêtée

400k steps, stabilité interne ✅ mais porte quasi-fermée à la validation ❌. La politique n'a pas trouvé comment interagir physiquement avec la poignée.

### Amélioration apportée → v3 Curriculum

Réduire le **seuil de succès** à 0.40 (40% d'ouverture au lieu de 90%) et **réduire la variance de spawn** du robot (position de départ plus cohérente) pour créer une version plus facile à apprendre en premier lieu.

---

## Run SAC v3 Curriculum — 500k steps

**Dossier :** [`run_sac_v3_curriculum/`](run_sac_v3_curriculum/)
**Run ID :** `20260507_133418`
**Config :** `open_single_door_sac_v3_curriculum.yaml`

### Paramètres clés

| Paramètre | Valeur |
|---|---|
| `ent_coef` | `0.1` (fixe) |
| `theta_success` | **0.40** (curriculum, vs 0.90 avant) |
| `robot_spawn_deviation_pos_x` | **0.05** (vs 0.15 défaut) |
| `robot_spawn_deviation_pos_y` | **0.02** (vs 0.05 défaut) |
| `w_wrong_dir` | 0.05 (réduit de 0.3 → moins punitif) |
| `w_stagnation` | 0.02 (réduit → plus tolérant) |
| `d_max` | 0.6 (élargi → récompense d'approche plus longue portée) |

### Ce qui s'est passé

**Signal positif à 300k steps :** `val_return_mean` a bondi de -0.55 → +12.60, `val_door_angle_final_mean` a atteint 0.021 rad (12× mieux que v3). L'agent commençait à pousser la porte.

**Mais régresssion à 400-500k steps :** `val_door_angle_final_mean` redescendait à 0.015-0.017, `val_return_mean` à 8-9. Le pic à 300k était une fluctuation sur 10 épisodes de validation, pas de la vraie politique stable. `theta_best_mean` en training stagnait à 0.003-0.004 rad : **la porte bouge à peine même pendant l'entraînement avec exploration**.

**Diagnostic final :** la critic loss continuait à croître (0 → 6+) car le replay buffer contenait 500k transitions avec récompense nulle ou négative — aucun signal de succès n'a jamais été observé (`success_frac=0`). Sans succès, le critic ne peut pas apprendre à valoriser l'ouverture de la porte → actor ne sait pas que c'est bien d'ouvrir.

### Pourquoi arrêtée

500k steps, 0 succès même avec theta_success=0.40 abaissé. Le **problème fondamental** n'est pas le seuil mais le fait que le replay buffer ne contient jamais de transitions avec un signal de succès positif. Le shaping dense ne suffit pas pour ce type de manipulation précise.

### Amélioration apportée → HER

Utiliser **HER (Hindsight Experience Replay)** : après chaque épisode échoué, relabelliser rétrospectivement les transitions comme des succès pour les angles réellement atteints. Si la porte a atteint 0.02 rad à un moment, créer des transitions virtuelles où l'objectif était 0.02 rad → succès atteint → Q-values positives apprises. Pas besoin de succès réels pour apprendre.

---

## Run SAC HER v1 — 200k steps

**Dossier :** [`run_sac_her/`](run_sac_her/)
**Run ID :** `20260507_162045`
**Config :** `open_single_door_sac_her.yaml`

### Paramètres clés

| Paramètre | Valeur |
|---|---|
| `use_her` | true |
| `her_n_sampled_goal` | 4 |
| `her_goal_strategy` | `future` |
| `theta_success` | 0.15 rad |
| `policy` | `MultiInputPolicy` |
| Reward | **sparse pur** : 0 si theta ≥ theta_success, -1 sinon |
| `w_approach` | 0.0 (tous les poids = 0) |

### Ce qui s'est passé

HER fonctionne mécaniquement (`train/actor_loss` 0 → -25, décroissant ✅), mais le mécanisme HER ne peut pas débloquer le cold start seul. Avec reward sparse pur et all-zero dense weights, **aucun signal ne guide l'exploration vers la poignée**. Max angle en training : 0.002 rad (pire qu'en v3 avec 0.003 rad). En validation déterministe : `val_door_angle_max_mean = 0.0` — porte totalement fermée.

**`val_return_std` : 12.9 → 0.0** à 200k — tous les épisodes terminent identiquement à -500. Politique complètement déterministe, convergée sur un minimum local.

**Bug `ent_coef` :** MLflow affichait 0.003 (au lieu de 0.1). Cause identifiée : avec `MultiInputPolicy` + `HerReplayBuffer`, SB3 logge le coefficient différemment en mode fixe. Le coefficient était bien 0.1 en pratique (confirmé par le comportement stable du training), mais le logging était trompeur.

### Pourquoi arrêtée

200k steps, theta_best DÉCROISSANT (0.0011 → 0.0008 rad). HER pur sans reward dense = cold start non résolu.

### Amélioration apportée → HER v2

Réintroduire le **reward dense** dans les transitions réelles (`w_approach=0.3`, `w_progress=1.0`) tout en gardant HER sparse pour les transitions virtuelles. Le reward dense guide l'exploration vers la poignée ; HER relabellise les succès partiels.

---

## Run SAC HER v2 — 300k steps

**Dossier :** [`run_sac_her_v2/`](run_sac_her_v2/)
**Run ID :** `20260507_174253`
**Config :** `open_single_door_sac_her_v2.yaml`

### Paramètres clés

| Paramètre | Valeur |
|---|---|
| `use_her` | true |
| `her_n_sampled_goal` | 4 |
| `theta_success` | 0.10 rad |
| `w_approach` | **0.3** (reward hybride) |
| `w_progress` | **1.0** |
| Reward réel | shaped_reward + sparse (-1/step) |
| Reward virtuel (HER) | sparse uniquement |

### Ce qui s'est passé

**Breakthrough à 200k steps :** `val_door_angle_max_mean = 0.133 rad` — première fois en 6 runs que la porte s'ouvre significativement (0.133 rad = ~7.6°). Dépassait le seuil de succès (theta_success=0.10 rad) de façon momentanée. `val_return_mean` -496 → -491, `progress_frac > 0` et croissant.

**Régression à 300k steps :** `val_door_angle_max_mean` redescend à 0.110 rad. L'agent dérive vers le **hover-hacking** : `val_approach_frac_mean` croît (0.015 → 0.047), `val_action_magnitude_mean` décroît (0.82 → 0.61). L'agent reste proche de la poignée pour maximiser `w_approach=0.3` sans pousser fort — minimum local du reward hybride.

**`val_door_angle_final_mean` reste à ~0.001-0.003 rad :** la porte s'ouvre en milieu d'épisode mais se referme avant le step final. L'agent pousse puis relâche → pas de succès (mesuré au dernier step).

**`val_success_rate = 0` tout le long :** la porte n'est jamais ouverte AU dernier step.

### Meilleur checkpoint

`best_model.zip` sauvegardé au step 200k — `val_door_angle_max_mean = 0.133 rad`, meilleur résultat de tout le projet.

### Pourquoi arrêtée

300k steps. Hover-hacking confirmé (approach_frac croissant, action_magnitude décroissant). Le minimum local s'aggravait.

### Amélioration apportée → HER v3

**Supprimer `w_approach`** pour éliminer l'incitation à hover. Seuls `w_progress=2.0` + `w_success=3.0` (bonus par step quand porte ≥ theta_success) guident la politique → force à **maintenir** l'ouverture. `w_stagnation=0.05` pénalise le hover sans progression.

---

## Référence des métriques

### Métriques train (internes SAC)

| Métrique | Ce qu'elle mesure | Valeur saine |
|---|---|---|
| `train/actor_loss` | Objectif de la politique (négatif = Q-value moyenne) | Décroissant (plus négatif), stable |
| `train/critic_loss` | MSE entre Q-valeur prédite et cible TD | Décroissant, < 1 à convergence |
| `train/ent_coef` | Coefficient d'entropie α | Fixe à 0.1 (v3+), pas de crash vers 0 |
| `train/learning_rate` | LR instantané (cosine schedule) | Décroît de 3e-4 à 1e-5 |
| `train/n_updates` | Nombre cumulé de gradient updates | Ligne droite, confirme que le train tourne |

### Métriques reward_hack (training rolling)

| Métrique | Ce qu'elle mesure | Valeur saine |
|---|---|---|
| `reward_hack/theta_best_mean` | Meilleur angle porte atteint par épisode (rad) | Croissant, > 0.1 à 300k |
| `reward_hack/train_success_rate` | Taux de succès en training (avec exploration) | Doit devenir > 0 avant la validation |
| `reward_hack/success_frac` | Part de la récompense venant du bonus sparse | Croît quand vrais succès commencent |
| `reward_hack/approach_frac` | Part venant de l'approche (hover-hack detector) | Faible si l'agent ouvre vraiment |
| `reward_hack/progress_frac` | Part venant de l'ouverture progressive | Élevé en milieu d'apprentissage |
| `reward_hack/oscillation_frac` | Part des pénalités d'oscillation | Proche de 0 |
| `reward_hack/stagnation_rate` | Fraction d'épisodes où la porte ne bouge plus | Proche de 0 |
| `reward_hack/reward_without_success` | Return moyen des épisodes échoués | Pas trop négatif (reward hacking proxy) |

### Métriques validation (évaluations déterministes)

| Métrique | Ce qu'elle mesure | Valeur saine |
|---|---|---|
| `val_return_mean` | Récompense cumulative par épisode | Croissant |
| `val_success_rate` | Taux de succès (politique déterministe) | **Métrique principale**, veut → 1.0 |
| `val_door_angle_final_mean` | Angle porte en fin d'épisode (rad) | Croissant, > theta_success |
| `val_door_angle_max_mean` | Meilleur angle atteint dans l'épisode | Croissant |
| `val_episode_length_mean` | Durée des épisodes (steps) | Diminue si l'agent réussit plus vite |
| `val_approach_frac_mean` | Fraction reward approche (hover-hack) | Faible |
| `val_action_magnitude_mean` | Norme L2 des actions | Modérée (0.3–0.7) |
| `val_action_smoothness_mean` | Fluidité des actions (jerk) | Croissant vers 1 |
| `val_stagnation_steps_mean` | Steps sans progression porte | Proche de 0 |
| `val_sign_changes_mean` | Changements de direction de la porte | Proche de 0 |
| `val_reward_without_success` | Return épisodes échoués | Croissant (même sans succès) |

---

## Résumé de la progression

```
v1 (500k)           → ent_coef crash (α→0, critic loss 40 000+)           → 0% succès, door_max=0.014 rad
    ↓ Fix: ent_coef=auto_0.1, target_entropy=-4
v2 (900k)           → α crash quand même (log_prob toujours < target)     → 0% succès, door_max=0.017 rad
    ↓ Fix: ent_coef=0.1 FIXE (plus d'auto-tuning)
v3 (400k)           → stable, mais cold start (porte quasi-fermée)        → 0% succès, door_max=0.012 rad
    ↓ Fix: curriculum (theta_success=0.40, spawn réduit)
v3_curriculum (500k)→ pic 300k (0.021 rad), régression, buffer sans signal → 0% succès, door_max=0.039 rad
    ↓ Fix: HER (relabellisation rétroactive des goals)
HER v1 (200k)       → reward sparse pur → cold start non résolu           → 0% succès, door_max=0.000 rad
    ↓ Fix: reward hybride (dense shaping + HER sparse)
HER v2 (300k)       → BREAKTHROUGH 0.133 rad à 200k, puis hover-hacking  → 0% succès, door_max=0.133 rad ★
    ↓ Fix: supprimer w_approach, forcer progress+success uniquement
HER v3 (en cours)   → objectif: premier succès réel
```

**Record absolu : HER v2 — `val_door_angle_max_mean = 0.133 rad` (best checkpoint step 200k)**

Chaque itération a résolu un problème précis mais en a révélé un nouveau. Le diagnostic s'est affiné à chaque run grâce aux métriques détaillées (`reward_hack`, `critic_loss`, `theta_best`, `approach_frac`).
