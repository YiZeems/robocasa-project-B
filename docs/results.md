# Résultats expérimentaux

> **Statut** : cette page est mise à jour après chaque run complet.  
> Les valeurs `[À compléter]` sont des placeholders — ils seront remplacés avec les résultats réels.

---

## 1. Résumé des meilleurs runs

| Run | Algo | Steps | Val. Success | Test Success | Best Step | Durée (h) |
|---|---|---:|---:|---:|---:|---:|
| SAC debug (seed=0) | SAC | 300k | [À compléter] | — | [À compléter] | ~1 |
| SAC principal (seed=0) | SAC | 3M | [À compléter] | [À compléter] | [À compléter] | ~19 |
| SAC tuned (seed=0) | SAC | 2M | [À compléter] | [À compléter] | [À compléter] | ~13 |
| PPO baseline (seed=0) | PPO | 5M | [À compléter] | [À compléter] | [À compléter] | ~30 |

**Commande pour reproduire ce tableau depuis les fichiers JSON :**

```bash
for f in outputs/*/train_summary.json; do
  echo "=== $f ==="
  python -c "
import json
d = json.load(open('$f'))
print(f'  best_val_success: {d.get(\"best_validation_success\", \"?\"):.3f}')
print(f'  best_step:        {d.get(\"best_validation_step\", \"?\")}')
print(f'  run_id:           {d.get(\"run_id\", \"?\")}')
"
done
```

---

## 2. Métriques anti-hacking

| Run | `approach_frac` | `stagnation_steps` | `sign_changes` | `door_angle_max` |
|---|---:|---:|---:|---:|
| SAC debug | [À compléter] | [À compléter] | [À compléter] | [À compléter] |
| SAC principal | [À compléter] | [À compléter] | [À compléter] | [À compléter] |
| SAC tuned | [À compléter] | [À compléter] | [À compléter] | [À compléter] |
| PPO baseline | [À compléter] | [À compléter] | [À compléter] | [À compléter] |

**Interprétation cible :**
- `approach_frac` < 0.3 → pas de hover hacking
- `stagnation_steps` < 20 → pas de blocage chronique
- `sign_changes` < 5 → pas d'oscillation pathologique
- `door_angle_max` > 0.8 → agent approche de la solution

---

## 3. Comparaison best checkpoint vs final checkpoint

| Run | Success @best | Success @final | Écart | Interprétation |
|---|---:|---:|---|---|
| SAC principal | [À compléter] | [À compléter] | [À compléter] | [À compléter] |
| PPO baseline | [À compléter] | [À compléter] | [À compléter] | [À compléter] |

> Un écart `best >> final` indique du surentraînement après le best checkpoint.  
> Pour le rapport, toujours reporter les métriques du **best checkpoint**.

---

## 4. Courbes d'apprentissage

> Générer avec : `make plot PLOT_RUNS="outputs/OpenCabinet_SAC_seed0_*/" SMOOTH=3`  
> Fichiers : `outputs/plots/`

### 4.1 Taux de succès validation

`outputs/plots/success_rate.png` — [Insérer l'image après le run final]

**Éléments à commenter :**
- Step de premier succès
- Vitesse de convergence
- Présence de plateau
- Step du best checkpoint

### 4.2 Grille récapitulative (summary)

`outputs/plots/summary.png` — [Insérer l'image après le run final]

### 4.3 Métriques anti-hacking

`outputs/plots/anti_hacking.png` — [Insérer l'image après le run final]

**Éléments à commenter :**
- Évolution de `approach_frac` — décroît-elle après les premiers épisodes ?
- `stagnation_steps` — reste-t-il faible tout au long ?
- `sign_changes` — présence ou absence d'oscillation ?

---

## 5. Analyse vidéo

> Générer avec : `make eval-video CONFIG=... CHECKPOINT=... EPISODES=20`  
> Fichier : `outputs/eval/videos/<run_id>_best_episode_<ts>.mp4`

### 5.1 Meilleur épisode (SAC principal)

| Propriété | Valeur |
|---|---|
| Episode ID | [À compléter] |
| Seed | [À compléter] |
| Success | [À compléter] |
| θ final | [À compléter] |
| θ max | [À compléter] |
| Longueur | [À compléter] steps |
| Score | [À compléter] |

**Description comportementale :** [À compléter après visionnage de la vidéo]

Exemples d'éléments à observer :
- Le robot approche-t-il directement de la poignée ?
- Y a-t-il des oscillations visibles ?
- La porte est-elle ouverte progressivement ou d'un coup ?
- Le robot reste-t-il immobile à un moment (stagnation) ?

### 5.2 Pire épisode (échec type)

| Propriété | Valeur |
|---|---|
| Episode ID | [À compléter] |
| Seed | [À compléter] |
| θ final | [À compléter] |
| Score | [À compléter] |

**Analyse de l'échec :** [À compléter]

---

## 6. Limites des résultats

### Limites observées

| Limite | Impact | Piste d'amélioration |
|---|---|---|
| Seed unique (seed=0) | Variance inconnue | Multi-seeds (3–5 seeds) |
| Pas de sim-to-real | Résultats valides en simulation uniquement | Domain randomization |
| Observations d'état uniquement | Sensible aux changements de positions initiales | Ajouter RGB |
| Navigation exclue | Robot toujours face au placard | Intégrer base mobile |

### Échecs observés

[À compléter après analyse des vidéos et des métriques]

---

## 7. Éléments encore à compléter

- [ ] Lancer SAC principal (3M steps, ~19h)
- [ ] Lancer PPO baseline (5M steps, ~30h)
- [ ] Exécuter `make eval-test` sur les deux best checkpoints
- [ ] Générer les vidéos : `make eval-video`
- [ ] Générer les courbes : `make plot`
- [ ] Remplir les tableaux de métriques ci-dessus
- [ ] Insérer les images dans les sections "Courbes d'apprentissage"
- [ ] Décrire le comportement observé dans les vidéos
- [ ] Analyser l'écart best vs final (surentraînement)
- [ ] Comparer SAC vs PPO sur toutes les métriques

---

## 8. Recommandations Git pour les artefacts

Les fichiers suivants peuvent être versionnés dans Git (taille raisonnable) :

```bash
# Courbes PNG (~200–500 Ko chacune)
git add outputs/plots/
git commit -m "Add training curves for SAC principal run"

# JSON de résumé (~10 Ko)
git add outputs/<run_id>/train_summary.json
git commit -m "Add SAC principal run summary"
```

Les fichiers suivants **ne doivent pas** être versionnés (taille trop grande) :

```text
checkpoints/    ← ~50 Mo par checkpoint
outputs/eval/videos/  ← ~50–200 Mo par vidéo MP4
mlruns/         ← potentiellement plusieurs Go
```

Pour partager les checkpoints et vidéos : utiliser Google Drive ou Hugging Face Hub et documenter les liens dans `docs/results.md`.
