# Méthodologie, reward et métriques

Ce document regroupe le cadre méthodologique RL, le reward shaping, les métriques, les limites et les pistes d’amélioration.


---

## Méthodes utilisées

### 1) Cadre expérimental

- Domaine: apprentissage par renforcement pour manipulation robotique en simulation (RoboCasa).
- Tâche de référence: `OpenCabinet` (ou alias pédagogiques `OpenSingleDoor`, `OpenDoor`).
- Robot: `PandaOmron`.
- Objectif d'agent: maximiser la récompense cumulée et le taux de succès.

### 2) Méthode RL retenue

#### Algorithme principal: SAC (Stable-Baselines3)

Raisons:
- off-policy et plus sample-efficient sur contrôle continu,
- bonne adéquation avec la manipulation continue RoboCasa,
- intégration stable dans SB3 avec replay buffer et reprise par checkpoints.

Baseline comparative:
- PPO est conservé comme baseline on-policy via `configs/train/open_single_door_ppo_baseline.yaml`.

Hyperparamètres configurés dans `configs/train/*.yaml`:
- SAC: `learning_rate`, `buffer_size`, `learning_starts`, `batch_size`, `tau`, `gamma`, `train_freq`, `gradient_steps`, `ent_coef`.
- PPO: `learning_rate`, `n_steps`, `batch_size`, `gamma`, `gae_lambda`, `clip_range`, `ent_coef`, `vf_coef`, `n_epochs`.

### 3) Représentation observation/action

#### Action space
- directement issu de `raw_env.action_spec`, converti en `gymnasium.spaces.Box(float32)`.

#### Observation space
- cas 1: wrapper gym robosuite stable (`GymWrapper`) -> observation native.
- cas 2 (fallback nominal sur cette stack): observation dict flattenée en vecteur 1D via `gymnasium.spaces.utils.flatten`.

Le fallback garantit la compatibilité SB3 quand le `GymWrapper` renvoie une shape d'observation instable entre resets.

### 4) Détection du succès

Fonction: `robocasa_telecom.utils.success.infer_success`.

Stratégie hiérarchique:
1. lire `info["success"]`, `info["task_success"]` ou `info["is_success"]`,
2. sinon sonder `_check_success()` sur `raw_env`,
3. sinon remonter la chaîne de wrappers (`env.env...`) jusqu'à 10 niveaux.

Intérêt:
- rendre l'évaluation robuste malgré les variations d'API entre wrappers.

### 5) Pipeline entraînement

Implémentation: `robocasa_telecom/rl/train.py`.

Étapes:
1. chargement config train + env,
2. création de l'environnement,
3. instrumentation Monitor (`monitor.csv`),
4. construction PPO,
5. construction PPO ou SAC,
6. entraînement avec checkpoints périodiques et validation,
7. sauvegarde modèle final + métadonnées de reprise,
8. mini-évaluation post-train,
9. export `training_curve.csv`, `validation_curve.csv` et `train_summary.json`.

### 6) Pipeline évaluation

Implémentation: `robocasa_telecom/rl/evaluate.py`.

Étapes:
1. chargement config + checkpoint,
2. rollouts sur `num_episodes`,
3. agrégation `return_mean`, `return_std`, `success_rate`, `episode_length_*`, `action_magnitude_*`, `door_angle_final_*` si disponible,
4. export JSON horodaté.

### 7) Reproductibilité

Mécanismes appliqués:
- seed explicite pour train/eval,
- commits figés robocasa/robosuite dans le script de setup,
- reprise possible depuis checkpoints périodiques + replay buffer SAC,
- auto-resume du dernier run incomplet compatible,
- sauvegarde du résumé JSON par run,
- convention de nommage incluant seed et timestamp.

### 8) Méthode cluster SLURM

- `train_array.sbatch`: parallélisation par seeds via `SLURM_ARRAY_TASK_ID`.
- `eval.sbatch`: évaluation d'un checkpoint unique avec export métriques.
- wrappers shell uniformisent local/cluster avec les mêmes interfaces.

### 9) Limites connues

- performance dépendante des assets et de la version MuJoCo/driver GPU,
- warnings non bloquants `mink`, `mimicgen` et `gym` hérités des dépendances amont,
- pas de benchmark multi-tâches à ce stade.

### 10) Extensions recommandées

- ajouter normalisation d'observation et benchmarking multi-seeds automatisé,
- ajouter comparaison image-based si la deadline le permet,
- ajouter suite de benchmarks multi-tâches RoboCasa.


---

## Reward Shaping — Documentation détaillée

> Implémentation : `robocasa_telecom/envs/reward.py`
> Configuration : `configs/env/open_single_door.yaml` (section `reward:`)

---

### 1. Motivation

La reward native de RoboCasa pour la tâche `OpenCabinet` est principalement une
**reward dense d'approche** complétée par un **bonus sparse de succès**. Ce design
simple produit systématiquement du reward hacking en pratique :

> L'agent apprend à rester immobile devant la poignée de porte (à ~5 cm) et
> accumule la reward d'approche sans jamais ouvrir la porte.
> Taux de succès : 0 %. Reward cumulative : 200–400.

Ce phénomène est connu sous le nom de **hover hacking** ou **approach reward
gaming**. Il est particulièrement fréquent sur les tâches de manipulation où :

- La reward d'approche est dense (signal à chaque step)
- Le bonus de succès est rare et difficile à atteindre au début

**Lien cours :** Reward shaping (Ng et al., 1999) — le façonnage de récompense
peut introduire un biais si les composantes ajoutées ne préservent pas l'invariance
par potential shaping. Le défi est de guider l'exploration sans créer de
comportements parasites exploitables.

---

### 2. Problèmes identifiés avec les rewards naïves

| Problème | Mécanisme | Symptôme observable |
|---|---|---|
| **Hover hacking** | Reward d'approche sans condition de progression | `approach_frac > 0.5`, `success_rate ≈ 0` |
| **Oscillation** | Aller-retour sur la porte = reward de progression nette | `sign_changes_mean` élevé |
| **Stagnation** | Rester près de la poignée sans agir | `stagnation_steps_mean` élevé |
| **Actions parasites** | Actions fortes récompensées si elles ouvrent légèrement | `action_magnitude_mean` élevé |
| **Wrong direction** | Refermer légèrement la porte ne coûte rien | `door_angle_delta` oscillant |

---

### 3. Formule complète

```
R_t = w_approach    × r_approach(d_t, θ_t)
    + w_progress    × r_progress(θ_t)
    + w_success     × r_success(θ_t)
    + w_action_reg  × r_action_reg(a_t)
    + w_stagnation  × r_stagnation(d_t, ctr_t)
    + w_wrong_dir   × r_wrong_dir(θ_t)
    + w_oscillation × r_oscillation(Δθ_window_t)
```

#### Définitions des composantes

**r_approach(d, θ) :**

```
r_approach = clip(1 - d / d_max, 0, 1)   si θ < θ_success
           = 0                             si θ ≥ θ_success
```

Guide le robot vers la poignée. Désactivé une fois la porte ouverte (gating).

**r_progress(θ) :**

```
δ = max(0, θ - θ_best)
r_progress = δ / θ_success
θ_best ← max(θ_best, θ)   [high-watermark]
```

Signal principal. Ne vaut > 0 que si l'agent bat son propre record d'ouverture.

**r_success(θ) :**

```
r_success = 1   si θ ≥ θ_success = 0.90
          = 0   sinon
```

Bonus sparse dominant. Reçu à chaque step où la porte est ouverte ≥ 90 %.

**r_action_reg(a) :**

```
r_action_reg = -‖a‖²   (norme L2 au carré)
```

Pénalise les actions fortes et saccadées.

**r_stagnation(d, ctr) — conditionnel :**

```
r_stagnation = -1   si d < d_prox ET ctr ≥ stagnation_n
             = 0    sinon
ctr s'incrémente quand θ ne progresse pas ; se reset quand θ progresse.
```

**r_wrong_dir(θ) :**

```
r_wrong_dir = -(θ_best - θ)   si θ < θ_best - tol_wrong_dir
            = 0                sinon
```

Pénalise le recul de la porte par rapport au meilleur angle atteint.

**r_oscillation(Δθ_window) :**

```
sign_changes = nombre de fois où signe(Δθ_i) ≠ signe(Δθ_{i-1}) sur la fenêtre
r_oscillation = -sign_changes / oscillation_window
              si sign_changes ≥ oscillation_threshold ET δ_progress < 1e-5
r_oscillation = 0 sinon
```

---

### 4. Table des coefficients et hyperparamètres

| Paramètre | Valeur par défaut | Rôle | Risque si trop élevé | Risque si trop faible |
|---|---:|---|---|---|
| `w_approach` | 0.05 | Poids reward approche | Hover hacking | Exploration lente |
| `w_progress` | 1.0 | Poids signal principal | — | Apprentissage lent |
| `w_success` | 5.0 | Poids bonus succès | — | Agent ignore la tâche |
| `w_action_reg` | 0.01 | Poids régularisation action | Actions trop timides | Jerk excessif |
| `w_stagnation` | 0.05 | Poids pénalité stagnation | Over-exploration | Stagnation non pénalisée |
| `w_wrong_dir` | 0.3 | Poids pénalité recul | Politique trop prudente (v3 : potentiellement trop punitif tôt) | Porte repoussée librement |
| `w_oscillation` | 0.2 | Poids pénalité oscillation | Politique figée | Oscillation non corrigée |
| `theta_success` | 0.90 | Seuil de succès normalisé | — | Tâche trop facile/difficile |
| `d_max` | 0.5 m | Portée max reward approche | — | — |
| `d_prox` | 0.12 m | Seuil "proche de la poignée" | Faux positifs stagnation | Pénalité trop tardive |
| `stagnation_n` | 50 | Steps avant déclenchement stagnation | Faux positifs fréquents | Hover hacking long |
| `tol_wrong_dir` | 0.02 | Tolérance avant pénalité recul | — | — |
| `oscillation_window` | 20 | Fenêtre de détection oscillation | Détection lente | Faux positifs sur micro-oscillations |
| `oscillation_threshold` | 4 | Changements de signe min pour déclencher | Faux positifs | Oscillation non détectée |

---

### 5. Propriétés du design

#### 5.1 Dominance du succès

```
R_max_approche = w_approach × 1.0 × 500 steps = 0.05 × 500 = 25
R_max_success  = w_success  × 1.0 × 500 steps = 5.0 × 500 = 2500
```

Le bonus de succès est **100× plus rentable** que la reward d'approche maximale.
L'agent a toujours intérêt à terminer la tâche plutôt qu'à stagner devant la poignée.

#### 5.2 Impossibilité d'osciller pour du profit

Avec le high-watermark : `r_progress(θ) = max(0, θ - θ_best)`. Si l'agent oscille
la porte entre 0.3 et 0.4, il ne reçoit de reward de progression **que la première
fois** qu'il atteint 0.4. Les aller-retours suivants donnent 0. De plus, si
l'oscillation est détectée par la fenêtre glissante, une pénalité supplémentaire
s'applique.

#### 5.3 Conditionnalité de la stagnation

La pénalité de stagnation ne s'active que si **les deux conditions sont vraies
simultanément** :

- `d_ee_handle < 0.12 m` (robot déjà proche de la poignée)
- `ctr >= 50 steps` (sans progression récente)

Cela évite de pénaliser le robot lors de la phase d'exploration initiale où il
cherche encore la poignée depuis loin.

#### 5.4 Approche gated

La reward d'approche est **nulle dès que la porte est ouverte** (θ ≥ 0.90). L'agent
ne peut pas augmenter la reward d'approche en continuant à s'agiter devant une porte
déjà ouverte.

---

### 6. Problèmes observés en pratique

#### 6.1 Hover hacking — observé sur le run debug

**Symptômes :** `return_mean ≈ 200–400`, `val_success_rate = 0%`,
`approach_frac > 0.7`. L'agent parque l'effecteur à ~5 cm de la poignée et reste
immobile.

**Cause :** Le signal de reward d'approche est dense et facilement exploitable.
Le bonus de succès est trop rare pour être découvert pendant l'exploration initiale.

**Fix appliqué :** High-watermark progress + gating approche + pénalité stagnation
(50 steps) + dominance du bonus succès (100×). Ces quatre mécanismes combinés ont
éliminé le hover hacking sur le run debug.

#### 6.2 Politique trop contrainte par w_wrong_dir (risque v3)

**Problème potentiel :** `w_wrong_dir=0.3` pénalise tout recul de la porte par
rapport au meilleur angle atteint. Tôt dans l'entraînement, quand l'agent n'a
jamais ouvert la porte au-delà de 0.05 rad, un petit recul de 0.02 rad suffit à
déclencher la pénalité. Cela peut inhiber les mouvements d'exploration nécessaires
pour trouver le contact correct.

**Mitigation :** Surveiller `val_door_angle_max_mean` sur les premières 200k steps
de v3. Si < 0.1, réduire `w_wrong_dir` à 0.1 et augmenter `tol_wrong_dir` à 0.05.

#### 6.3 oscillation_frac négatif dans MLflow

**Bug détecté :** La fraction `oscillation_frac` apparaissait négative dans les
courbes MLflow.

**Cause :** La pénalité d'oscillation est une reward négative. Divisée par
`abs(total_mean)` sans `abs()` au numérateur, la fraction devenait négative.

**Fix :** `abs()` appliqué au numérateur. Statut : résolu.

---

### 7. Table récapitulative pour le rapport

| Composante | Description | Coefficient | Risque évité | Métrique de vérification |
|---|---|---:|---|---|
| approach | Distance EEF → poignée normalisée | 0.05 | Hover hacking (gating activé) | `approach_frac_mean < 0.3` |
| progress | Δθ high-watermark (uniquement si nouveau max) | 1.0 | Oscillation pour du profit | `door_angle_max_mean > 0.5` |
| success | Bonus sparse θ ≥ 0.90 | 5.0 | Ignorer la tâche | `success_rate > 0` |
| action_reg | −‖a‖² | 0.01 | Actions parasites et jerk | `action_smoothness_mean` stable |
| stagnation | −1 si proche+bloqué ≥ 50 steps | 0.05 | Hover hacking long | `stagnation_steps_mean < 20` |
| wrong_dir | −Δθ_négatif si recul > tolérance | 0.3 | Refermer la porte | `sign_changes_mean < 5` |
| oscillation | −sign_changes/window | 0.2 | Oscillation rapide | `sign_changes_mean < 10` |

---

### 8. Métriques de vérification (MLflow)

Loggées dans `train/reward_hack/*` et `validation/*` :

| Métrique MLflow | Interprétation | Seuil d'alerte |
|---|---|---|
| `train/reward_hack/approach_frac` | Fraction reward due à l'approche | > 0.5 |
| `train/reward_hack/stagnation_steps_mean` | Steps stagnants par épisode | > 100 |
| `train/reward_hack/oscillation_steps_mean` | Steps oscillants par épisode | > 50 |
| `train/reward_hack/sign_changes_mean` | Changements de signe Δθ | > 15 |
| `val_approach_frac_mean` | Même signal sur validation | > 0.5 |
| `val_door_angle_max_mean` | Best door angle par épisode | < 0.3 (agent bloqué) |
| `reward_without_success` | Return moyen des épisodes en échec | Élevé + succès nul |

---

### 9. Exemples de logs attendus

#### Comportement sain (pas de reward hacking)

```
validation — step 500000:
  val_success_rate          = 0.54
  val_approach_frac_mean    = 0.18   ← faible (pas de hover hacking)
  val_stagnation_steps_mean = 12.3   ← faible
  val_sign_changes_mean     = 3.1    ← faible (pas d'oscillation)
  val_door_angle_max_mean   = 0.87   ← haut (robot approche de la solution)
```

#### Comportement de hover hacking (à corriger)

```
validation — step 100000:
  val_success_rate          = 0.00
  val_approach_frac_mean    = 0.72   ← ALERTE hover hacking
  val_stagnation_steps_mean = 187    ← ALERTE blocage chronique
  val_sign_changes_mean     = 2.1
  val_door_angle_max_mean   = 0.12   ← robot à peine sorti
  reward_without_success    = 185.3  ← reward haute sans succès
```

---

### 10. Ablations recommandées

Pour un rapport plus rigoureux, les ablations suivantes permettent de quantifier
l'impact de chaque composante :

| Ablation | Config | Hypothèse |
|---|---|---|
| Sans pénalité stagnation (`w_stagnation=0`) | Modifier `reward:` dans le YAML | `stagnation_steps_mean` augmente, `success_rate` baisse |
| Sans gating approche (toujours actif) | Modifier `r_approach` dans `reward.py` | `approach_frac` augmente, hover hacking réapparaît |
| Sans high-watermark (`r_progress = θ_t`) | Modifier `reward.py` | `sign_changes_mean` augmente, oscillation profitable |
| Sans pénalité oscillation (`w_oscillation=0`) | Modifier `reward:` dans le YAML | `sign_changes_mean` augmente |

---

### 11. Modifier la reward

Tous les coefficients sont exposés dans `configs/env/open_single_door.yaml` sous
la clé `reward:` et rechargés automatiquement par `EnvConfig`. Aucune modification
du code Python n'est nécessaire pour changer les coefficients.

```yaml
reward:
  w_approach: 0.05
  w_progress: 1.0
  w_success: 5.0
  w_action_reg: 0.01
  w_stagnation: 0.05
  w_wrong_dir: 0.3
  w_oscillation: 0.2
  theta_success: 0.90
  d_max: 0.5
  d_prox: 0.12
  stagnation_n: 50
  tol_wrong_dir: 0.02
  oscillation_window: 20
  oscillation_threshold: 4
```

---

### 12. Lien cours — MDP et reward shaping

La tâche est modélisée comme un **MDP** (Markov Decision Process) :

- **S** : vecteur d'observation 220D (positions articulaires, vitesses, distance
  EEF-poignée, angle de porte, pose de l'effecteur)
- **A** : espace d'action 12D continu, contrôle OSC (Operational Space Control)
- **R** : reward shapée R_t décrite ci-dessus
- **γ** : facteur de discount = 0.99 (horizon effectif ~100 steps)
- **T** : épisode de 500 steps maximum

La reward shapée R_t est une **transformation de la reward originale** (sparse
bonus succès uniquement). Elle vise à résoudre le problème de **credit assignment**
sur des épisodes de 500 steps : sans signal dense, le gradient de la politique
ne peut pas attribuer la récompense aux actions initiales (approche de la poignée)
qui conditionnent le succès final.

Le risque théorique (Ng et al., 1999) est que les composantes non-potential
(stagnation, wrong_dir, oscillation) modifient la politique optimale du MDP. Ce
risque est mitigé en pratique par la dominance écrasante du bonus de succès (100×
la reward d'approche), qui garantit que la politique optimale reste "ouvrir la porte"
plutôt que "éviter les pénalités".


---

## Métriques — Référence complète

> Toutes les métriques sont loggées dans MLflow (`mlruns/`).
> Les métriques de validation sont aussi dans `outputs/<run_id>/validation_curve.csv`.

---

### 1. Métriques de performance

#### `val_success_rate` / `success_rate`

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

#### `val_return_mean` / `return_mean`

**Définition :** return cumulé moyen par épisode sur N épisodes de validation.

**Pourquoi :** indicateur de progrès global. Devrait croître au fil du training et se stabiliser.

**Interpréter :**
- Croissance monotone → bonne convergence
- Plateau soudain → early stopping probable
- Croissance sans `success_rate` → reward hacking

**Erreurs révélées :** return élevé + succès nul = indicateur quasi-certain de reward gaming.

**Dans MLflow :** `validation/val_return_mean`

---

#### `val_episode_length_mean`

**Définition :** longueur moyenne des épisodes (en steps) sur validation.

**Pourquoi :** une longueur décroissante indique que l'agent réussit plus vite (ou échoue plus vite). Une longueur constante à 500 indique que l'agent épuise l'horizon sans succès.

**Interpréter :**
- Décroissante → agent plus efficace
- Constante à 500 → agent échoue systématiquement

**Dans MLflow :** `validation/val_episode_length_mean`

---

#### `val_door_angle_final_mean`

**Définition :** angle de porte normalisé [0, 1] à la **fin** de l'épisode, moyenné sur N épisodes.

**Pourquoi :** mesure l'ouverture atteinte en moyenne à la fin de l'épisode. Plus fiable que `success_rate` seul pour distinguer "0 % d'ouverture" de "80 % d'ouverture sans succès".

**Interpréter :**
- < 0.2 → agent bloqué loin de la solution
- 0.5–0.8 → progression mais pas de succès consistant
- > 0.9 → succès ou très proche

**Erreurs révélées :** si `door_angle_final_mean` ≈ 0.05 après 1M steps → hover hacking ou mauvaise initialisation.

**Dans MLflow :** `validation/val_door_angle_final_mean`

---

#### `val_door_angle_max_mean`

**Définition :** angle de porte maximal atteint **à un moment quelconque** de l'épisode (high-watermark), moyenné sur N épisodes.

**Pourquoi :** révèle si l'agent approche de la solution sans maintenir l'ouverture. Si `door_angle_max_mean` est élevé mais `door_angle_final_mean` est faible → l'agent ouvre puis referme.

**Interpréter :**
- `max >> final` → oscillation / refermeture active
- `max ≈ final` → l'agent maintient l'ouverture (bon signe)
- `max < 0.3` → agent bloqué

**Dans MLflow :** `validation/val_door_angle_max_mean`

---

#### `val_door_angle_delta`

**Définition :** `door_angle_final − door_angle_initial` par épisode.

**Pourquoi :** mesure la progression nette. Élimine les biais dus à une initialisation à un angle déjà partiellement ouvert.

**Dans MLflow :** `eval_video/door_angle_delta` (dans le debug CSV)

---

#### `time_to_success`

**Définition :** step auquel l'agent atteint θ ≥ 0.90 pour la première fois dans l'épisode. `-1` si pas de succès.

**Pourquoi :** mesure l'efficacité temporelle. Un agent qui réussit au step 50 est meilleur qu'un agent qui réussit au step 490 (même horizon de 500).

**Dans MLflow :** `eval_video/time_to_success` (dans le debug CSV)

---

### 2. Métriques anti-hacking

#### `val_approach_frac_mean`

**Définition :** `approach_total / max(reward_total, ε)` — fraction de la reward totale due à la composante d'approche.

**Pourquoi :** indicateur direct de hover hacking. Si l'agent gagne principalement de la reward en restant proche de la poignée plutôt qu'en ouvrant la porte, cette fraction est élevée.

**Interpréter :**
- < 0.2 → sain (reward dominée par progress + success)
- 0.3–0.5 → à surveiller
- > 0.5 → alerte hover hacking

**Dans MLflow :** `validation/val_approach_frac_mean`

---

#### `val_stagnation_steps_mean`

**Définition :** nombre moyen de steps par épisode où le robot est près de la poignée (d < 0.12 m) sans progresser (ctr ≥ 50).

**Pourquoi :** mesure la tendance à stagner devant la poignée sans agir.

**Interpréter :**
- < 10 → sain
- 50–100 → l'agent passe du temps à stagner
- > 100 → hover hacking chronique

**Dans MLflow :** `validation/val_stagnation_steps_mean`

---

#### `val_sign_changes_mean`

**Définition :** nombre moyen de changements de signe de Δθ par épisode (fenêtre glissante de 20 steps).

**Pourquoi :** mesure l'oscillation de la porte. Un robot qui ouvre et referme alternativement produit beaucoup de changements de signe.

**Interpréter :**
- < 5 → sain
- 5–15 → oscillation modérée
- > 15 → oscillation pathologique

**Dans MLflow :** `validation/val_sign_changes_mean`

---

#### `reward_without_success`

**Définition :** return moyen des épisodes en échec (success=False).

**Pourquoi :** si cette valeur est élevée (proche du return global), l'agent accumule de la reward même sans accomplir la tâche — indicateur de reward gaming.

**Interpréter :**
- Proche de 0 → les épisodes ratés rapportent peu (bon signe)
- Élevé + `success_rate` faible → gaming sévère

**Dans MLflow :** `validation/val_reward_without_success`

---

### 3. Métriques de qualité des actions

#### `val_action_smoothness_mean`

**Définition :** `mean(‖a_t − a_{t-1}‖)` sur l'épisode. Mesure la jerkiness (saccades).

**Pourquoi :** une politique apprise avec régularisation action (`w_action_reg > 0`) devrait produire des actions lisses. Des actions saccadées peuvent indiquer une policy instable ou une reward mal balancée.

**Interpréter :**
- Faible et stable → politique fluide
- Croissant → la politique devient plus agressive (potentiellement mauvais signe)

**Dans MLflow :** `validation/val_action_smoothness_mean`

---

#### `val_action_magnitude_mean`

**Définition :** `mean(‖a_t‖)` par épisode.

**Pourquoi :** complète `action_smoothness`. Des actions de grande magnitude peuvent indiquer que l'agent "pousse fort" sans finesse.

**Dans MLflow :** `validation/val_action_magnitude_mean`

---

### 4. Métriques d'algorithme

#### SAC

| Métrique | Nom MLflow | Interprétation |
|---|---|---|
| Perte acteur | `train/actor_loss` | Devrait être négatif et décroissant |
| Perte critique (Q1+Q2) | `train/critic_loss` | Devrait converger |
| Perte entropie | `train/entropy_loss` | Suit l'évolution du coefficient ent_coef |
| Valeur Q moyenne | `train/q_values` | Doit croître avec l'apprentissage |
| Coefficient entropie | `train/ent_coef` | Doit converger vers une valeur stable |

#### PPO

| Métrique | Nom MLflow | Interprétation |
|---|---|---|
| KL approx. | `train/approx_kl` | Doit rester < 0.05 (si > 0.1 → step trop grand) |
| Clip fraction | `train/clip_fraction` | Doit rester < 0.3 (si > 0.5 → clip_range trop petit) |
| Perte valeur | `train/value_loss` | Décroissante |
| Perte politique | `train/policy_gradient_loss` | Négatif |
| Perte entropie | `train/entropy_loss` | Légèrement négatif |

---

### 5. Métriques computationnelles

#### `train/fps`

**Définition :** steps d'environnement par seconde.

**Valeurs typiques :**
- SAC, 12 workers, MPS (Apple Silicon) : ~2 600 steps/min
- SAC, 12 workers, RTX 4070 : ~3 500 steps/min
- PPO, 1 worker, CPU : ~800 steps/min

**Interprétation :** une chute soudaine de FPS peut indiquer un goulot d'étranglement CPU (trop de gradient steps) ou un problème mémoire.

---

### 6. Métriques de reproductibilité

| Métrique | Où la trouver | Usage |
|---|---|---|
| `seed` | `train_summary.json` | Permet de reproduire exactement |
| `best_validation_step` | `train_summary.json` | Step du best_model.zip |
| `best_validation_success` | `train_summary.json` | Success rate du best checkpoint |
| `total_timesteps_done` | `train_summary.json` | Nombre de steps réellement effectués |
| `run_id` | Convention `task_algo_seed_timestamp` | Identifiant unique du run |

---

### 7. Localisation dans le code

| Métrique | Fichier source | Ligne/fonction |
|---|---|---|
| `summarize_rollout_episodes` | `utils/metrics.py` | Calcule toutes les métriques de rollout |
| `success_confidence_interval` | `utils/metrics.py` | Wilson 95% CI |
| `ValidationCallback._log_metrics` | `rl/train.py` | Log vers MLflow + CSV |
| `RewardHackMonitorCallback` | `rl/train.py` | Métriques intra-épisode anti-hacking |
| `AntiHackingReward.episode_summary` | `envs/reward.py` | Compteurs par épisode |

---

### 8. `validation_curve.csv` — colonnes

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


---

## Issues and Limitations

> This document covers both **technical issues encountered and resolved** during development, and **inherent limitations** of the current project design that would require future work to address.

---

### 1. Technical Issues Encountered and Resolved

#### Summary Table

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

#### 1.1 Ent_coef Auto-tuning Crash (SAC v1 et v2)

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

#### 1.2 Critic Loss Spikes

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

#### 1.3 Hover Hacking (Reward Gaming)

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

#### 1.4–1.16 Voir le tableau récapitulatif ci-dessus.

---

#### 1.17 Cold Start — Aucun Contact avec la Poignée (v3)

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

#### 1.18 Pic Transitoire sans Convergence (v3_curriculum)

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

#### 1.19 Replay Buffer Sans Signal Positif

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

#### 1.20 HER : reference_obs_space Dict Transmis aux Workers

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

### 2. Limitations Related to Method Choices

#### Summary Table

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

#### 2.1 Reward Shaping May Bias Learning

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

#### 2.2 SAC Requires Significant Tuning

SAC comporte plus d'hyperparamètres que PPO : `learning_rate`, `buffer_size`,
`batch_size`, `tau`, `gradient_steps`, `ent_coef`, `learning_starts`,
`target_update_interval`. La configuration actuelle est basée sur les pratiques
de la littérature et les diagnostics des runs v1/v2/v3, mais pas sur une recherche
systématique. La sensibilité aux hyperparamètres est inconnue.

---

#### 2.3 MuJoCo Simulation Does Not Capture the Real World

MuJoCo ne modélise pas : variabilité de friction, bruit de capteurs, jeu mécanique
des joints, ni l'apparence visuelle. Une politique entraînée en simulation
échouerait sur un vrai PandaOmron sans domain randomization et adaptation.

**Impact :** Les résultats sont valides uniquement pour l'environnement de
simulation. Aucune conclusion sur les performances en conditions réelles ne peut
être tirée.

---

#### 2.4 Contact Learning Is Inherently Difficult

La saisie de la poignée (~5 cm de diamètre) requiert un contact précis avec
application de force dans la bonne direction. En exploration aléatoire, les
contacts utiles sont très rares. Ce problème est intrinsèquement sample-inefficient.

**Lien cours :** Credit assignment — les récompenses tardives (ouverture de porte)
sont difficiles à attribuer aux actions initiales (approche de la poignée) sur
des horizons de 500 steps.

---

#### 2.5 12 Workers Complicate Debugging

Un crash dans un worker peut ralentir silencieusement l'entraînement sans erreur
dans le processus principal. Les logs de 12 workers sont entrelacés. La RAM est
amplifiée 12×.

---

#### 2.6 Single Seed — Unknown Variance

Avec un seul seed par algorithme, il est impossible de distinguer entre "SAC est
meilleur que PPO sur cette tâche" et "ce seed SAC a eu de la chance". L'intervalle
de confiance Wilson sur N épisodes au sein d'un même run ne quantifie que la
variance intra-run, pas la variance inter-seed.

**Mitigation dans le rapport :** présenter les résultats comme observations de
cas unique ; comparer les tendances qualitatives ; ne pas formuler de conclusions
statistiques fortes.

---

#### 2.7 No Ablations on Reward Components

Le reward shaping a 7 composantes et ~14 hyperparamètres. Sans ablations, il est
impossible de savoir quelles composantes contribuent à prévenir le reward hacking
et lesquelles sont redondantes ou nuisibles.

---

#### 2.8 Navigation Excluded

La base mobile est fixée. Le robot est toujours positionné face au placard. En
déploiement réel, le robot devrait naviguer depuis une position arbitraire, ce qui
interagit avec la tâche de manipulation.

---

#### 2.9 Compute Constraints

| Contrainte | Impact |
|---|---|
| Un seul GPU WSL2 (RTX 4070 ou équivalent) | Un seul seed par algorithme |
| Deadline : 7 mai 2026 | Pas d'ablations, pas de multi-seed, curriculum exclu |
| RAM WSL2 (même après fix à 56 GB) | Impossible de lancer plusieurs runs longs en parallèle |

---

#### 2.10 Toutes les Runs Abandonnées — 0% Success Rate

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

### 3. Known Warnings (Non-Blocking)

Les warnings suivants apparaissent à l'initialisation de l'environnement et sont
**attendus et sans impact fonctionnel** :

```
WARNING: mimicgen environments not imported since mimicgen is not installed!
WARNING: mink environments not imported since mink is not installed!
UserWarning: WARN: Gym has been unmaintained since 0.26...
```

Ces warnings proviennent de dépendances optionnelles de RoboCasa et n'affectent
pas la tâche `OpenCabinet` ni aucune fonctionnalité utilisée dans ce projet.


---

## Model Improvements and Future Work

> This document identifies concrete improvements to the current project, each linked to a concept from the IA705 — Learning for Robotics course.

---

### 1. Improvements Directly Linked to Course Concepts

#### Summary Table

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

#### 1.1 Fix ent_coef : Valeur Fixe (v3) — Critique, Fait

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

#### 1.2 State-Dependent Exploration — use_sde=True (v3)

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

#### 1.3 Observation Normalisation (VecNormalize)

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

#### 1.4 Curriculum Learning — Fait, Insuffisant (v3_curriculum)

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

#### 1.4b HER (Hindsight Experience Replay) — En cours

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

#### 1.5 Behavioral Cloning Pre-training + SAC Fine-tuning

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
## Schéma conceptuel
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

#### 1.6 Ablations sur les Composantes Reward

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

#### 1.7 Multi-Seed Evaluation

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

#### 1.8 Comparaison TD3

**État actuel :** SAC (principal) + PPO (baseline) implémentés. Pas de TD3.

**Amélioration :** Ajouter TD3 comme troisième algorithme :

```yaml
## configs/train/open_single_door_td3.yaml
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

#### 1.9 Replay Buffer Inspection

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

#### 1.10 Critère de Succès Soutenu

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

#### 1.11 Policy Evaluation on Held-Out Configurations

**État actuel :** Le split test utilise `seed=20000` qui randomise l'apparence de
la scène, mais pas le type de placard ni la plage de positions initiales.

**Amélioration :** Définir des ensembles d'évaluation réellement hors distribution :
- Styles de placard différents (non vus pendant l'entraînement)
- Distances robot-placard différentes
- Conditions d'éclairage différentes

**Lien cours :** Généralisation en robot learning — une politique qui fonctionne
uniquement pour la distribution d'entraînement n'est pas utile en déploiement.

---

### 2. Roadmap Prioritaire

#### Fait (avant deadline)

- `ent_coef=0.1` fixe (v3) — résout le crash d'auto-tuning de v1/v2
- `use_sde=True`, `sde_sample_freq=64` (v3) — exploration cohérente
- `gradient_steps=4`, `tau=0.01` (v2/v3) — stabilité du critic
- LR cosine schedule 3e-4 → 1e-5 (v3)
- Curriculum learning : `theta_success=0.40`, spawn déviation réduite (v3_curriculum)
- HER : `GoalConditionedWrapper` + `HerReplayBuffer`, `theta_success=0.15` (en cours)

#### Post-deadline prioritaire

- Observation normalisation (`VecNormalize`) — faible effort, fort impact
- Multi-seed (3–5 seeds) — résultats statistiquement valides
- Ablations reward — quantifier l'utilité de chaque composante

#### Long terme

- BC pre-training + SAC fine-tuning — résout définitivement le cold start
- Évaluation hors distribution — test de généralisation réelle
- Intégration navigation + base mobile — tâche complète PandaOmron
