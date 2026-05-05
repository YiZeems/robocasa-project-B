# Checklist de rendu — IA705, Telecom Paris

> Deadline : **jeudi 7 mai 2026 à 23h59**  
> Cocher chaque item avant le rendu.

---

## Vérifications techniques (code)

- [ ] `make check` passe sans erreur
- [ ] `make sanity` passe sans erreur (20 steps reset/step)
- [ ] Smoke test : `uv run python -m robocasa_telecom.train --config configs/train/open_single_door_sac_debug.yaml --seed 0 --total-timesteps 10 --no-auto-resume`
- [ ] Pas de fichier sensible ou chemin absolu personnel dans Git
- [ ] `.gitignore` couvre `checkpoints/`, `outputs/`, `mlruns/`, `logs/`, `.venv/`
- [ ] Aucune clé API, token ou secret dans le code

---

## Entraînement

- [ ] SAC debug (300k) terminé — `outputs/OpenCabinet_SAC_seed0_*/` existe
- [ ] SAC principal (3M) terminé — `checkpoints/<run_id>/best_model.zip` existe
- [ ] PPO baseline (5M) terminé — `checkpoints/<run_id_ppo>/best_model.zip` existe
- [ ] SAC tuned (2M) terminé (optionnel — si temps disponible)
- [ ] `train_summary.json` présent pour chaque run complété
- [ ] `validation_curve.csv` présent pour chaque run complété

---

## Évaluation

- [ ] `make eval-test` exécuté sur SAC principal `best_model.zip` — résultats dans `outputs/eval/`
- [ ] `make eval-test` exécuté sur PPO baseline `best_model.zip` — résultats dans `outputs/eval/`
- [ ] `test_success_rate` reporté dans `docs/results.md`
- [ ] Écart train/validation/test analysé et documenté

---

## Vidéos

- [ ] `make eval-video` exécuté sur SAC principal (20 épisodes, seed=0)
- [ ] `<run_id>_best_episode_*.mp4` existe et est lisible
- [ ] `<run_id>_worst_episode_*.mp4` existe (pour analyse des échecs)
- [ ] Vidéo décrite dans `docs/results.md` (comportement observé)
- [ ] Vidéo loggée dans MLflow (artefact `videos/`)

---

## Métriques et courbes

- [ ] `make plot` exécuté — `outputs/plots/` contient `summary.png`, `success_rate.png`, `door_angle.png`, `anti_hacking.png`
- [ ] Courbes insérées ou référencées dans `docs/results.md`
- [ ] Métriques anti-hacking vérifiées (`approach_frac < 0.5`, `stagnation < 100`)
- [ ] Tableau comparatif SAC vs PPO rempli dans `docs/results.md`
- [ ] Best checkpoint vs final checkpoint comparé

---

## MLflow

- [ ] `uv run mlflow ui --backend-store-uri ./mlruns` fonctionne
- [ ] Runs SAC et PPO visibles dans l'interface MLflow
- [ ] Métriques `val_success_rate`, `val_approach_frac_mean` présentes
- [ ] Screenshots ou export MLflow joint au rapport si nécessaire

---

## Documentation

- [ ] `README.md` complet et à jour
- [ ] `docs/project_report.md` rédigé et tous les `[À compléter]` remplacés
- [ ] `docs/results.md` rempli avec les valeurs réelles
- [ ] `docs/reward_shaping.md` présent
- [ ] `docs/metrics.md` présent
- [ ] `docs/video_generation.md` présent
- [ ] `docs/reproducibility.md` présent
- [ ] `docs/troubleshooting.md` présent

---

## Rapport académique

- [ ] Introduction : contexte, objectif, question de recherche
- [ ] Tâche : description OpenCabinet, robot PandaOmron, horizon, observation, action
- [ ] Formulation MDP : états, actions, reward, γ, succès/échec
- [ ] Méthodes : SAC (principal), PPO (baseline), justification des choix
- [ ] Reward shaping : formule complète, coefficients, mécanismes anti-hacking
- [ ] Protocole : n_envs, seeds, eval_freq, n_eval_episodes, split validation/test
- [ ] Résultats : tableaux + courbes + vidéos
- [ ] Discussion : SAC vs PPO, convergence, surentraînement
- [ ] Limites : seed unique, pas de sim-to-real, navigation exclue
- [ ] Perspectives : multi-seeds, curriculum, imitation learning, RGB
- [ ] Conclusion : réponse à la question de recherche
- [ ] Références : RoboCasa, SAC, PPO, SB3, RoboSuite, MuJoCo

---

## Git et GitHub

- [ ] Tous les commits pushés sur `main`
- [ ] Pas de fichier `mlruns/`, `checkpoints/`, `.venv/` dans Git
- [ ] Courbes PNG dans `outputs/plots/` committées (optionnel mais utile)
- [ ] `train_summary.json` des runs finaux committés (optionnel)
- [ ] README visible et bien rendu sur GitHub

---

## Ablations (optionnel — renforce le rapport)

- [ ] Run sans pénalité stagnation (`w_stagnation=0`) pour montrer l'effet
- [ ] Run avec reward naive (sans anti-hacking) pour montrer le hover hacking
- [ ] Comparaison `approach_frac` avant vs après le reward shaping

---

## Éléments encore à produire (mise à jour le 5 mai 2026)

| Élément | Statut | Action |
|---|---|---|
| SAC debug 300k | ✅ Terminé | — |
| SAC principal 3M | ⏳ En cours | Lancer `make train-sac SEED=0` |
| PPO baseline 5M | ⏳ À lancer | Lancer `make train-ppo-baseline SEED=0` |
| Eval test SAC | ⏳ Après run | `make eval-test CONFIG=... CHECKPOINT=...` |
| Eval test PPO | ⏳ Après run | `make eval-test CONFIG=... CHECKPOINT=...` |
| Vidéos | ⏳ Après run | `make eval-video CONFIG=... CHECKPOINT=...` |
| Courbes PNG | ⏳ Après run | `make plot PLOT_RUNS=...` |
| docs/results.md rempli | ⏳ Après runs | Remplir les placeholders |
| project_report.md finalisé | ⏳ Mercredi 6 mai | Rédaction |

---

## Recommandations finales

1. **Utiliser `best_model.zip`**, jamais `final_model.zip`, pour les résultats du rapport.
2. **Vérifier `approach_frac`** avant de conclure que le reward shaping fonctionne.
3. **Comparer best vs final** pour documenter le surentraînement.
4. **Citer les métriques anti-hacking** dans le rapport — c'est un apport méthodologique fort.
5. **Inclure une vidéo** : les examinateurs apprécient voir le comportement réel.
