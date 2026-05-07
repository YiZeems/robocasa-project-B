# Résultats expérimentaux

> **Statut** : mise à jour après les 4 premiers runs complets (v1, v2, v3, v3_curriculum).  
> Le run HER est en cours — ses métriques seront ajoutées ici dès sa conclusion.

---

## 1. Résumé des runs

| Run | Algo | Steps | Val. Success | `door_angle_final` (rad) | Problème diagnostiqué | Arrêtée |
|---|---|---:|---:|---:|---|---|
| SAC v1 | SAC | 500k | **0%** | 0.000 | `ent_coef` crash α→0 (critic loss 40k+) | Oui |
| SAC v2 | SAC | 900k | **0%** | 0.000 | `ent_coef` auto-tuning inévitablement α→0 | Oui |
| SAC v3 | SAC | 400k | **0%** | 0.004 | Cold start — pas de contact poignée | Oui |
| SAC v3 Curriculum | SAC | 500k | **0%** | 0.021 (pic 300k) | Buffer sans signal de succès | Oui |
| SAC HER | SAC+HER | en cours | — | — | — | Non |

> Toutes les métriques reportées ici correspondent au **checkpoint final** (pas de best checkpoint car succès = 0 dans tous les runs).

---

## 2. Métriques clés par run

### 2.1 Métriques validation (déterministiques)

| Run | `val_return_mean` | `door_angle_final` (rad) | `door_angle_max` (rad) | `val_success_rate` |
|---|---:|---:|---:|---:|
| SAC v1 (500k) | ~ -3.5 | 0.000 | 0.014 | 0% |
| SAC v2 (900k) | ~ -2.8 | 0.000 | 0.017 | 0% |
| SAC v3 (400k) | ~ -1.2 | 0.004 | 0.012 | 0% |
| SAC v3 Curriculum (500k) | +12.6 (pic 300k) | 0.021 (pic 300k) | 0.039 | 0% |

### 2.2 Métriques train internes

| Run | `ent_coef` final | `critic_loss` max | `actor_loss` final | `theta_best_mean` (rad) |
|---|---:|---:|---:|---:|
| SAC v1 (500k) | 0.009 (crash α→0) | ~48 000 | -37.5 | ~0.003 |
| SAC v2 (900k) | 0.001 (crash α→0) | ~116 828 | +7.2 (diverge) | ~0.003 |
| SAC v3 (400k) | 0.100 (stable) | 37.7 | -47.1 | ~0.003 |
| SAC v3 Curriculum (500k) | 0.100 (stable) | 10.3 | -48.2 | ~0.004 |

### 2.3 Métriques anti-hacking (validation)

| Run | `approach_frac` | `stagnation_steps` | `sign_changes` | Hover hacking ? |
|---|---:|---:|---:|---|
| SAC v1 (500k) | élevé (~0.6) | élevé (>200) | élevé (>10) | Oui (probable) |
| SAC v2 (900k) | élevé (~0.5) | élevé (>200) | élevé (>10) | Oui (probable) |
| SAC v3 (400k) | ~0.4 | ~180 | ~8 | Partiel |
| SAC v3 Curriculum (500k) | ~0.3 | ~150 | ~6 | Partiel |

> **Interprétation cible :** `approach_frac` < 0.3, `stagnation_steps` < 20, `sign_changes` < 5, `door_angle_max` > 0.8.  
> Aucun run n'a atteint ces cibles — la porte ne s'est jamais ouverte significativement.

---

## 3. Diagnostic des échecs par run

### SAC v1 — Crash entropie (ent_coef auto-tuning)

**Symptôme :** α (ent_coef) chute de 0.1 → 0.009 en 200k steps. Critic loss explose à ~48 000.

**Cause :** L'auto-tuning SAC résout `∇α = α × (-log_prob - target_entropy)`. Avec un espace d'action 12D gaussien, `log_prob ≈ -20`. Avec `target_entropy = -12` (auto), le gradient est toujours `-20 - (-12) = -8 < 0` → α → 0 sans jamais s'arrêter.

**Conséquence :** Sans entropie, la politique devient déterministe avant d'avoir appris. Les Q-values divergent (critic loss 40k+). L'agent reste statique.

**Fix v2 :** `ent_coef="auto_0.1"`, `target_entropy=-4`.

---

### SAC v2 — Même crash, juste retardé

**Symptôme :** α part de 0.1 et retombe à 0.001 en 100k steps (vs 200k en v1). Critic loss atteint 116 828 (pire que v1 sur 900k).

**Cause :** Avec `target_entropy=-4`, le gradient est `-20 - (-4) = -16 < 0`. Encore plus négatif qu'en v1. L'auto-tuning aggrave le problème en changeant `target_entropy` vers des valeurs moins négatives.

**Conséquence :** Actor loss remonte en territoire positif (+7.2) après 200k → politique anti-corrélée avec les Q-values → comportement aléatoire pur.

**Fix v3 :** `ent_coef=0.1` **fixe**, plus d'auto-tuning.

---

### SAC v3 — Cold start : pas de contact poignée

**Symptôme :** Toutes les métriques internes stables (α = 0.1 fixe ✅, critic_loss ~4–37 ✅, actor_loss stable à -47 ✅). Mais `door_angle_final = 0.004 rad` et `theta_best_mean ≈ 0.003 rad` en training.

**Cause :** Cold start problem. L'agent ne trouve pas comment établir le premier contact physique avec la poignée. Sans contact → reward de progression = 0 → Q-values neutres → pas de signal pour l'actor. Circulaire.

**Clé :** même avec exploration SDE active, la porte bouge de seulement 0.003 rad (moins d'un demi-degré) sur 400k steps.

**Fix curriculum :** `theta_success=0.40` (plus facile), spawn réduit (robot plus près de la poignée).

---

### SAC v3 Curriculum — Pic transitoire, pas de convergence

**Symptôme :** Signal positif à 300k (`val_return_mean` +12.6, `door_angle_final` 0.021 rad = 12× mieux que v3). Puis régression à 400–500k (retour à 0.015–0.017).

**Cause :** Le pic à 300k est une fluctuation statistique sur seulement 10 épisodes de validation. En training, `theta_best_mean` reste à 0.003–0.004 rad — la porte ne bouge pas significativement même avec exploration. Le replay buffer contient 500k transitions sans aucune récompense de succès (`success_frac = 0` tout le long).

**Conséquence :** Le critic ne peut pas apprendre que "ouvrir la porte est bien" car il n'a jamais observé un succès. La critic_loss croît (0 → 10.3) car les estimations de Q restent fausses faute de cible positive.

**Fix HER :** Hindsight Experience Replay — créer rétrospectivement des transitions virtuelles où l'objectif était l'angle réellement atteint dans l'épisode → Q-values positives même sans succès réel.

---

## 4. Courbes d'apprentissage

> Générées avec `python scripts/plot_runs.py` depuis la racine du projet.  
> Images stockées dans [`docs/courbes/`](courbes/).

### 4.1 Courbes par run

| Run | Dossier |
|---|---|
| SAC v1 | [`docs/courbes/run_sac_v1/`](courbes/run_sac_v1/) |
| SAC v2 | [`docs/courbes/run_sac_v2/`](courbes/run_sac_v2/) |
| SAC v3 | [`docs/courbes/run_sac_v3/`](courbes/run_sac_v3/) |
| SAC v3 Curriculum | [`docs/courbes/run_sac_v3_curriculum/`](courbes/run_sac_v3_curriculum/) |

### 4.2 Métriques clés à observer dans les courbes

- **`train/ent_coef`** : doit rester plat à 0.1 (v3+). Crash vers 0 = problème (v1, v2).
- **`train/critic_loss`** : doit décroître. Explosion = Q-values divergentes.
- **`train/actor_loss`** : doit devenir plus négatif (Q-values croissantes). Remontée positive = politique anti-corrélée.
- **`val_door_angle_final_mean`** : métrique principale. Doit croître vers `theta_success`. Stagne < 0.005 dans tous les runs.
- **`reward_hack/theta_best_mean`** : angle max atteint en training (avec exploration). Stagne à 0.003 → agent ne touche pas la poignée.

---

## 5. Limites des résultats actuels

| Limite | Impact | Résolution prévue |
|---|---|---|
| 0% succès dans tous les runs | Pas de comparaison SAC vs PPO possible | HER doit débloquer les premiers succès |
| Cold start non résolu (dense reward) | Reward shaping insuffisant pour manipulation précise | HER (sparse + relabellisation) |
| Seed unique (seed=0) | Variance inconnue | Multi-seeds après premier succès |
| Exploration SDE insuffisante | Agent ne trouve pas la poignée en 500k steps | HER + réduction spawn |
| theta_best_mean ≈ 0.003 rad | La porte ne bouge même pas en exploration | Problem fondamental → HER |

---

## 6. État en cours — SAC HER

**Config :** [`configs/train/open_single_door_sac_her.yaml`](../configs/train/open_single_door_sac_her.yaml)

**Principe :** Pour chaque épisode où la porte a atteint `θ_max`, créer 4 transitions virtuelles par step où le goal était un angle futur réellement atteint → reward sparse = 0 (succès virtuel). Le critic apprend que "ouvrir la porte à X rad est bien" même si X < 0.15 rad.

**Objectif :** Obtenir les premiers Q-values positives → d'abord `theta_best_mean > 0.01` en training → puis `val_success_rate > 0`.

Les métriques seront disponibles dans [`docs/courbes/run_sac_her/`](courbes/run_sac_her/) après 200k steps.

---

## 7. Recommandations Git pour les artefacts

Les fichiers suivants peuvent être versionnés dans Git :

```bash
# Courbes PNG (~200-500 Ko chacune)
git add docs/courbes/
git commit -m "Add training curves for all completed runs"

# Scripts de génération
git add scripts/plot_runs.py
```

Les fichiers suivants **ne doivent pas** être versionnés :

```text
checkpoints/           ← ~50 Mo par checkpoint
outputs/eval/videos/   ← ~50-200 Mo par vidéo MP4
mlruns/                ← potentiellement plusieurs Go
outputs/               ← données brutes (sauf train_summary.json)
```
