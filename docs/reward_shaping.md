# Reward Shaping — Documentation détaillée

> Implémentation : `robocasa_telecom/envs/reward.py`  
> Configuration : `configs/env/open_single_door.yaml` (section `reward:`)

---

## 1. Motivation

La reward native de RoboCasa pour la tâche `OpenCabinet` est principalement une **reward dense d'approche** complétée par un **bonus sparse de succès**. Ce design simple produit systématiquement du reward hacking en pratique :

> L'agent apprend à rester immobile devant la poignée de porte (à ~5 cm) et accumule la reward d'approche sans jamais ouvrir la porte. Taux de succès : 0 %. Reward cumulative : 200–400.

Ce phénomène est connu sous le nom de **hover hacking** ou **approach reward gaming**. Il est particulièrement fréquent sur les tâches de manipulation où :
- La reward d'approche est dense (signal à chaque step)
- Le bonus de succès est rare et difficile à atteindre au début

---

## 2. Problèmes identifiés avec les rewards naïves

| Problème | Mécanisme | Symptôme observable |
|---|---|---|
| **Hover hacking** | Reward d'approche sans condition de progression | `approach_frac > 0.5`, `success_rate ≈ 0` |
| **Oscillation** | Aller-retour sur la porte ≈ reward de progression nette | `sign_changes_mean` élevé |
| **Stagnation** | Rester près de la poignée sans agir | `stagnation_steps_mean` élevé |
| **Actions parasites** | Actions fortes récompensées si elles ouvrent légèrement | `action_magnitude_mean` élevé |
| **Wrong direction** | Refermer légèrement la porte ne coûte rien | `door_angle_delta` oscillant |

---

## 3. Formule complète

```
R_t = w_approach    × r_approach(d_t, θ_t)
    + w_progress    × r_progress(θ_t)
    + w_success     × r_success(θ_t)
    + w_action_reg  × r_action_reg(a_t)
    + w_stagnation  × r_stagnation(d_t, ctr_t)
    + w_wrong_dir   × r_wrong_dir(θ_t)
    + w_oscillation × r_oscillation(Δθ_window_t)
```

### Définitions des composantes

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

## 4. Table des coefficients et hyperparamètres

| Paramètre | Valeur par défaut | Rôle | Risque si trop élevé | Risque si trop faible |
|---|---:|---|---|---|
| `w_approach` | 0.05 | Poids reward approche | Hover hacking | Exploration lente |
| `w_progress` | 1.0 | Poids signal principal | — | Apprentissage lent |
| `w_success` | 5.0 | Poids bonus succès | — | Agent ignore la tâche |
| `w_action_reg` | 0.01 | Poids régularisation action | Actions trop timides | Jerk excessif |
| `w_stagnation` | 0.05 | Poids pénalité stagnation | Over-exploration | Stagnation non pénalisée |
| `w_wrong_dir` | 0.3 | Poids pénalité recul | Politique trop prudente | Porte repoussée librement |
| `w_oscillation` | 0.2 | Poids pénalité oscillation | Politique figée | Oscillation non corrigée |
| `theta_success` | 0.90 | Seuil de succès normalisé | — | Tâche trop facile/difficile |
| `d_max` | 0.5 m | Portée max reward approche | — | — |
| `d_prox` | 0.12 m | Seuil "proche de la poignée" | FP stagnation | Pénalité trop tardive |
| `stagnation_n` | 50 | Steps avant déclenchement stagnation | FP fréquents | Hover hacking long |
| `tol_wrong_dir` | 0.02 | Tolérance avant pénalité recul | — | — |
| `oscillation_window` | 20 | Fenêtre de détection oscillation | Détection lente | FP sur micro-oscillations |
| `oscillation_threshold` | 4 | Changements de signe min pour déclencher | Faux positifs | Oscillation non détectée |

---

## 5. Propriétés du design

### 5.1 Dominance du succès

```
R_max_approche = w_approach × 1.0 × 500 steps = 0.05 × 500 = 25
R_max_success  = w_success  × 1.0 × 500 steps = 5.0 × 500 = 2500
```

Le bonus de succès est **100× plus rentable** que la reward d'approche maximale. L'agent a toujours intérêt à terminer la tâche plutôt qu'à stagner.

### 5.2 Impossibilité d'osciller pour du profit

Avec le high-watermark : `r_progress(θ) = max(0, θ - θ_best)`. Si l'agent oscille la porte entre 0.3 et 0.4, il ne reçoit de reward de progression **que la première fois** qu'il atteint 0.4. Les aller-retours suivants donnent 0. De plus, si l'oscillation est détectée par la fenêtre glissante, une pénalité supplémentaire s'applique.

### 5.3 Conditionnalité de la stagnation

La pénalité de stagnation ne s'active que si **les deux conditions sont vraies simultanément** :
- `d_ee_handle < 0.12 m` (robot déjà proche de la poignée)
- `ctr >= 50 steps` (sans progression récente)

Cela évite de pénaliser le robot lors de la phase d'exploration initiale où il cherche encore la poignée depuis loin.

### 5.4 Approche gated

La reward d'approche est **nulle dès que la porte est ouverte** (θ ≥ 0.90). L'agent ne peut pas augmenter la reward d'approche en continuant à s'agiter devant une porte déjà ouverte.

---

## 6. Table récapitulative pour le rapport

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

## 7. Métriques de vérification (MLflow)

Loggées dans `train/reward_hack/*` et `validation/*` :

| Métrique MLflow | Interprétation | Seuil d'alerte |
|---|---|---|
| `train/reward_hack/approach_frac` | Fraction reward due à l'approche | > 0.5 |
| `train/reward_hack/stagnation_steps_mean` | Steps stagnants par épisode | > 100 |
| `train/reward_hack/oscillation_steps_mean` | Steps oscillants par épisode | > 50 |
| `train/reward_hack/sign_changes_mean` | Changements de signe Δθ | > 15 |
| `val_approach_frac_mean` | Même signal sur validation | > 0.5 |
| `val_door_angle_max_mean` | Best door angle par épisode | < 0.3 (bloqué) |
| `reward_without_success` | Return moyen des épisodes en échec | Élevé + succès nul |

---

## 8. Ablations recommandées

Pour un rapport plus rigoureux, les ablations suivantes permettent de quantifier l'impact de chaque composante :

| Ablation | Config | Hypothèse |
|---|---|---|
| Sans pénalité stagnation (`w_stagnation=0`) | Modifier `reward:` dans le YAML | `stagnation_steps_mean` augmente, `success_rate` baisse |
| Sans gating approche (toujours actif) | Modifier `r_approach` dans `reward.py` | `approach_frac` augmente, hover hacking réapparaît |
| Sans high-watermark (`r_progress = θ_t`) | Modifier `reward.py` | `sign_changes_mean` augmente, oscillation profitable |
| Sans pénalité oscillation (`w_oscillation=0`) | Modifier `reward:` dans le YAML | `sign_changes_mean` augmente |

---

## 9. Exemples de logs attendus

### Comportement sain (pas de reward hacking)

```
validation — step 500000:
  val_success_rate        = 0.54
  val_approach_frac_mean  = 0.18   ← faible (pas de hover hacking)
  val_stagnation_steps_mean = 12.3  ← faible
  val_sign_changes_mean   = 3.1    ← faible (pas d'oscillation)
  val_door_angle_max_mean = 0.87   ← haut (robot approche de la solution)
```

### Comportement de hover hacking (à corriger)

```
validation — step 100000:
  val_success_rate        = 0.00
  val_approach_frac_mean  = 0.72   ← ALERTE hover hacking
  val_stagnation_steps_mean = 187   ← ALERTE blocage chronique
  val_sign_changes_mean   = 2.1
  val_door_angle_max_mean = 0.12   ← robot à peine sorti
  reward_without_success  = 185.3  ← reward haute sans succès
```

---

## 10. Modifier la reward

Tous les coefficients sont exposés dans `configs/env/open_single_door.yaml` sous la clé `reward:` et rechargés automatiquement par `EnvConfig`. Aucune modification du code Python n'est nécessaire pour changer les coefficients.

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
