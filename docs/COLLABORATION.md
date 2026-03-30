# Collaboration Git (équipe de 4)

## 1) Branches de travail

Branches recommandées et déjà préparées:
- `main`: branche stable, démontrable, exécutable.
- `develop`: intégration continue d'équipe.
- `feature/member1-env-foundation`
- `feature/member2-training-pipeline`
- `feature/member3-eval-visualization`
- `feature/member4-slurm-repro`

## 2) Répartition conseillée des responsabilités

- Member 1 (`feature/member1-env-foundation`):
  - `robocasa_telecom/envs/`
  - configs environnement
  - robustesse setup et compatibilité tâches

- Member 2 (`feature/member2-training-pipeline`):
  - `robocasa_telecom/rl/train.py`
  - tuning hyperparamètres
  - export des métriques d'entraînement

- Member 3 (`feature/member3-eval-visualization`):
  - `robocasa_telecom/rl/evaluate.py`
  - `scripts/visualize_env.py`
  - scripts d'analyse et graphes

- Member 4 (`feature/member4-slurm-repro`):
  - `scripts/slurm/`
  - wrappers run local/cluster
  - reproductibilité, tests de job array

## 3) Workflow recommandé

1. Sync local:
   - `git checkout develop`
   - `git pull origin develop`

2. Travail feature:
   - `git checkout feature/memberX-...`
   - commits atomiques et ciblés.

3. Pull Request:
   - PR de `feature/*` vers `develop`.
   - review croisée (au moins 1 reviewer).

4. Promotion stable:
   - merge `develop` vers `main` seulement après validation exécutable.

## 4) Convention de commit

Format conseillé:
- `<scope>: <action concise>`

Scopes utilisés dans ce dépôt:
- `env`, `rl`, `tools`, `utils`, `scripts`, `slurm`, `docs`, `tests`

Exemples:
- `env: add robust OpenSingleDoor alias handling`
- `rl: export training curve from monitor csv`
- `slurm: map array index to explicit seed list`
- `docs: add file-by-file technical reference`

## 5) Checklist PR

Avant merge vers `develop`:
- code exécutable en local,
- script shell valide (`bash -n`),
- modules Python compilent (`python -m compileall`),
- test config minimal passe (`tests/test_config_loading.py`),
- doc mise à jour si interface ou comportement modifié.

## 6) Gestion des conflits

- Toujours rebase ou merge `develop` avant gros commit final.
- Ne jamais réécrire l'historique de `main`.
- En cas de conflit sur un fichier partagé (`README`, config train), faire une mini-réunion d'alignement avant résolution.
