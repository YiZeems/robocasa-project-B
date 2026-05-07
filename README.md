# RoboCasa Door Opening - IA705

Projet d'apprentissage par renforcement pour la tâche `OpenCabinet` de RoboCasa : apprendre à un robot PandaOmron à ouvrir une porte de placard en simulation MuJoCo.

Le dépôt contient le code d'entraînement, d'évaluation, de suivi MLflow et de génération vidéo. La documentation a été volontairement resserrée :

- [Rapport de projet](docs/project_report.md) : contexte, méthode, résultats, limites et discussion.
- [Courbes d'entraînement](docs/courbes/README.md) : analyse détaillée des runs SAC successifs.
- [Guide technique](docs/technical_guide.md) : architecture, runbook, plateformes, CI, vidéo, dépannage.
- [Méthodologie, reward et métriques](docs/methodology_reward_metrics.md) : méthodes RL, reward shaping, métriques, limites, améliorations.
- [Expériences et résultats](docs/experiments_results.md) : protocole, progression, résultats et diagnostics.
- [Reproductibilité et rendu](docs/reproducibility_submission.md) : reproduction, checklist, collaboration Git.
- [Analyse des courbes](docs/curves_analysis.md) : lecture des graphes, diagnostics run par run et synthèse.
- Ce README : installation, commandes utiles et repères pour lancer le projet.

## Résumé

| Élément | Choix retenu |
|---|---|
| Simulateur | RoboCasa + RoboSuite + MuJoCo |
| Tâche | `OpenCabinet` / ouverture d'une porte de placard |
| Robot | PandaOmron, base fixe, bras Panda contrôlé |
| Algorithme principal | SAC, Stable-Baselines3 |
| Baseline prévue | PPO |
| Observations | État bas niveau, pas d'images RGB |
| Suivi expérimental | MLflow, CSV, checkpoints, vidéos |
| Meilleur résultat actuel | SAC + HER v2, `door_angle_max = 0.133 rad` à 200k steps |

La métrique principale reste le taux de succès en validation. Dans les runs réalisés, le succès strict reste à 0 %, mais HER v2 est le premier run à produire une ouverture visible de la porte. Le rapport explique ce résultat sans le maquiller : le projet est surtout un diagnostic expérimental sur les difficultés d'exploration, de reward shaping et de stabilité de SAC.

## Structure utile

```text
robocasa-project-B/
├── configs/
│   ├── env/                         # configuration RoboCasa et reward
│   └── train/                       # configs SAC, PPO, HER
├── robocasa_telecom/
│   ├── envs/                        # adaptateur Gymnasium / RoboCasa, reward
│   ├── rl/                          # train, evaluate, eval_video
│   └── utils/                       # métriques, checkpoints, vidéo, I/O
├── scripts/                         # setup, lancement, plots, SLURM
├── tests/                           # smoke tests et tests utilitaires
├── docs/
│   ├── project_report.md            # rapport académique unique
│   ├── technical_guide.md           # guide technique regroupé
│   ├── methodology_reward_metrics.md
│   ├── experiments_results.md
│   ├── reproducibility_submission.md
│   ├── curves_analysis.md
│   └── courbes/README.md            # analyse des courbes
├── checkpoints/                     # modèles sauvegardés, non versionnés
├── outputs/                         # résultats, CSV, vidéos, non versionnés
└── mlruns/                          # tracking MLflow local
```

Les répertoires `external/robocasa` et `external/robosuite` sont installés par le script de setup.

## Installation

Pré-requis communs : Python 3.11, `uv`, `git`, MuJoCo compatible avec la plateforme.

```bash
git clone https://github.com/YiZeems/robocasa-project-B.git
cd robocasa-project-B
bash scripts/setup_uv.sh
```

Le script installe l'environnement virtuel, les dépendances Python, RoboCasa/RoboSuite et les assets nécessaires. Pour forcer le téléchargement et la vérification des assets :

```bash
DOWNLOAD_ASSETS=1 VERIFY_ASSETS=1 bash scripts/setup_uv.sh
```

Vérification rapide :

```bash
make check
make sanity
pytest tests/test_platform_smoke.py -v
```

## Entraînement

Commandes principales :

```bash
make train-sac-debug SEED=0
make train-sac SEED=0
make train-sac-tuned SEED=0
make train-ppo-baseline SEED=0
```

Commande directe équivalente :

```bash
uv run python -m robocasa_telecom.train \
  --config configs/train/open_single_door_sac.yaml \
  --seed 0
```

Pour démarrer un run propre sans reprise automatique :

```bash
uv run python -m robocasa_telecom.train \
  --config configs/train/open_single_door_sac.yaml \
  --seed 0 \
  --no-auto-resume
```

Les artefacts sont écrits dans :

```text
outputs/<run_id>/                 # CSV, résumés JSON, configs résolues
checkpoints/<run_id>/             # best_model.zip, final_model.zip, replay buffer
mlruns/                           # tracking MLflow
```

Pour le rapport et les vidéos, utiliser `best_model.zip`, pas `final_model.zip`.

## Évaluation

Validation :

```bash
make eval-validation \
  CONFIG=configs/train/open_single_door_sac.yaml \
  CHECKPOINT=checkpoints/<run_id>/best_model.zip \
  EPISODES=50
```

Test final sur seeds non vus :

```bash
make eval-test \
  CONFIG=configs/train/open_single_door_sac.yaml \
  CHECKPOINT=checkpoints/<run_id>/best_model.zip \
  EPISODES=50
```

Commande directe :

```bash
uv run python -m robocasa_telecom.evaluate \
  --config configs/train/open_single_door_sac.yaml \
  --checkpoint checkpoints/<run_id>/best_model.zip \
  --num-episodes 50 \
  --split test \
  --deterministic
```

## Vidéos

Le rendu vidéo est fait après entraînement, dans un environnement single-worker. Cela évite les problèmes OpenGL avec `SubprocVecEnv`.

```bash
make eval-video \
  CONFIG=configs/train/open_single_door_sac.yaml \
  CHECKPOINT=checkpoints/<run_id>/best_model.zip \
  EPISODES=20 \
  SEED=0
```

La sélection est faite en deux passes : scorer plusieurs épisodes sans rendu, puis rejouer le meilleur épisode avec le même seed. Les vidéos et métadonnées sont placées dans `outputs/eval/videos/`.

## MLflow et courbes

Lancer l'interface :

```bash
uv run mlflow ui --backend-store-uri ./mlruns
```

Puis ouvrir `http://127.0.0.1:5000`.

Générer des courbes :

```bash
make plot PLOT_RUNS="outputs/OpenCabinet_SAC_seed0_*/" SMOOTH=3
```

Comparer plusieurs runs :

```bash
uv run python scripts/plot_training.py \
  --run outputs/OpenCabinet_SAC_seed0_*/ outputs/OpenCabinet_PPO_seed0_*/ \
  --label "SAC seed0" "PPO seed0" \
  --out outputs/plots/
```

## Métriques à regarder

| Métrique | Lecture |
|---|---|
| `val_success_rate` | Critère principal : porte ouverte selon le seuil de succès |
| `val_door_angle_max_mean` | Meilleur angle atteint pendant l'épisode |
| `val_door_angle_final_mean` | Angle final, utile pour détecter une porte qui se referme |
| `val_approach_frac_mean` | Détecte le hover-hacking si la reward vient surtout de l'approche |
| `val_stagnation_steps_mean` | Temps passé proche de la poignée sans progrès |
| `train/ent_coef` | Santé de SAC ; le crash vers 0 a cassé v1 et v2 |
| `train/critic_loss` | Explosion = Q-values instables |
| `train/actor_loss` | Remontée vers le positif = politique qui se dégrade |

## Résultats actuels

| Run | Steps | Succès validation | `door_angle_max` | Diagnostic |
|---|---:|---:|---:|---|
| SAC v1 | 500k | 0 % | 0.014 rad | `ent_coef` auto-tuning s'effondre |
| SAC v2 | 900k | 0 % | 0.017 rad | même problème, retardé |
| SAC v3 | 400k | 0 % | 0.012 rad | stable mais cold start |
| SAC v3 curriculum | 500k | 0 % | 0.039 rad | petit pic, pas de convergence |
| SAC HER v1 | 200k | 0 % | 0.000 rad | HER sparse pur insuffisant |
| SAC HER v2 | 300k | 0 % | 0.133 rad | meilleure ouverture, puis hover-hacking |

Le run le plus intéressant est HER v2 à 200k steps : la porte bouge enfin de manière visible. La suite naturelle consiste à supprimer la reward d'approche trop attractive, renforcer le maintien de l'ouverture, puis relancer HER avec plusieurs seeds si un premier succès strict apparaît.

## Dépannage rapide

| Symptôme | Action |
|---|---|
| `.venv` absent ou cassé | `bash scripts/setup_uv.sh` |
| Assets RoboCasa manquants | `DOWNLOAD_ASSETS=1 VERIFY_ASSETS=1 bash scripts/setup_uv.sh` |
| MLflow port 5000 occupé | `uv run mlflow ui --backend-store-uri ./mlruns --port 5001` |
| Reprise automatique indésirable | ajouter `--no-auto-resume` |
| Vidéo noire | générer avec `eval_video.py`, pas depuis les workers d'entraînement |
| OOM avec 12 workers | réduire `n_envs` dans la config YAML |
| Warnings `mink`, `mimicgen`, `gym` | connus et non bloquants dans ce projet |

## Références

- RoboCasa : https://robocasa.ai
- RoboSuite : https://robosuite.ai
- SAC : Haarnoja et al., 2018, https://arxiv.org/abs/1801.01290
- PPO : Schulman et al., 2017, https://arxiv.org/abs/1707.06347
- Stable-Baselines3 : https://stable-baselines3.readthedocs.io
- MuJoCo : https://mujoco.org
