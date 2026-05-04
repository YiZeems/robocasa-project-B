# Packages

## Base

Le packaging du projet est désormais défini dans [`pyproject.toml`](../pyproject.toml).

## Dépendances Python directes

- `stable-baselines3`
- `gymnasium`
- `pyyaml`
- `numpy`
- `pandas`
- `tensorboard`
- `tqdm`
- `matplotlib`
- `imageio`
- `robosuite_models`

## Dépendances externes

- `robosuite`
- `robocasa`

## Dépendances système

- `git`
- `cmake`
- `ffmpeg`

## Remarque

Les anciens exports Conda/Pip restent disponibles dans `docs/packages/` à titre d'historique, mais ils ne sont plus la source de vérité pour l'installation.
