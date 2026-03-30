# Méthodes utilisées

## Problème

Apprentissage d'une tâche atomique RoboCasa avec `PandaOmron`.
Configuration par défaut: ouvrir une porte de meuble (`OpenCabinet`).

## Algorithme RL

- Algorithme: **PPO** (Stable-Baselines3)
- Observations: vecteur flatten (dict obs -> flat) via adapter Gymnasium
- Actions: espace continu natif de l'environnement

## Raisons du choix PPO

- robuste pour une baseline de projet
- simple à monitorer (TensorBoard + CSV)
- compatible avec parallélisation par seeds via SLURM array

## Pipeline

1. Chargement config train YAML
2. Création env RoboCasa via `envs.factory`
3. Wrapping Monitor SB3
4. Entraînement PPO
5. Sauvegarde checkpoints périodiques + modèle final
6. Évaluation post-train et export métriques

## Mesures exportées

Train:
- `training_curve.csv` (reward/length/time épisode)
- `train_summary.json` (timesteps + métriques eval rapides)

Eval:
- `eval_*.json` (return mean/std, success rate)

## Gestion du succès

Ordre de priorité:
1. lecture `info[success|task_success|is_success]`
2. fallback sur `_check_success()` du raw env
