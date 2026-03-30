# Packages installés et utilisés

Ce document répond à deux besoins:
1. **Liste complète des packages installés** dans l'environnement projet.
2. **Liste des packages utilisés par le code** du dépôt, avec leur rôle.

## 1) Inventaire complet installé (export exact)

Les exports complets ont été générés le **30 mars 2026** depuis l'environnement `robocasa_telecom_ci`.

- Conda export complet: `docs/packages/conda_list_export_2026-03-30.txt`
- Pip freeze complet: `docs/packages/pip_freeze_2026-03-30.txt`

Ces deux fichiers sont la référence exhaustive (versions incluses) de ce qui est installé.

## 2) Packages installés explicitement par le projet

### 2.1 Base Conda (`environment.yml`)

- `python=3.11`: runtime Python du projet.
- `pip`: installation des dépendances Python (dont robosuite/robocasa).
- `setuptools`, `wheel`: build et installation des packages Python.
- `git`: clone des dépôts externes `robosuite` et `robocasa`.
- `cmake`: compilation native de certaines dépendances.
- `ffmpeg`: backend vidéo (export/lecture vidéo dans stack imageio).
- `patchelf`: utilitaire de linking binaire (surtout utile en Linux cluster).

### 2.2 Dépendances pip directes (`requirements-project.txt`)

- `stable-baselines3`: algorithme PPO et API RL.
- `gymnasium`: interface environnement RL (`Env`, `spaces`, wrappers).
- `pyyaml`: lecture des fichiers de configuration YAML.
- `numpy`: calcul numérique, vecteurs d'observation/actions.
- `pandas`: manipulation de tableaux/CSV (analyse post-run).
- `tensorboard`: visualisation de l'entraînement.
- `tqdm`: barres de progression.
- `matplotlib`: tracés et visualisation de courbes.
- `imageio`: lecture/écriture image et vidéo.

### 2.3 Dépendances installées en editable par `scripts/setup_conda.sh`

- `robosuite` (commit `aaa8b9b214ce8e77e82926d677b4d61d55e577ab`): simulateur robotique de base.
- `robocasa` (commit `9a3a78680443734786c9784ab661413edb87067b`): tâches kitchen/manipulation au-dessus de robosuite.

## 3) Packages utilisés directement dans le code du dépôt

Cette section couvre les imports présents dans `robocasa_telecom/`, `scripts/` et `tests/`.

- `robocasa`: enregistrement des tâches RoboCasa dans robosuite.
- `robosuite`: création des environnements (`robosuite.make`), wrappers Gym.
- `stable_baselines3`: entraînement (`PPO`) et callbacks/checkpoints.
- `gymnasium`: classes `Env`, espaces (`Box`, `Dict`), flatten des observations.
- `numpy`: conversion/normalisation des observations et actions.
- `yaml` (`PyYAML`): chargement/écriture des configs.
- `matplotlib`, `pandas`, `imageio`, `tensorboard`, `tqdm`: utilisés par workflow data/visualisation/monitoring (et requis dans l'environnement projet).

## 4) Dépendances transitives importantes (indirectes)

Ces packages ne sont pas importés directement partout dans le dépôt, mais sont nécessaires au fonctionnement de la stack:

- `mujoco`: moteur physique sous-jacent.
- `glfw`, `PyOpenGL`: rendu et contexte graphique.
- `torch`: backend de policy networks pour SB3.
- `scipy`: calcul scientifique utilisé par dépendances RL/simulation.
- `pillow`, `imageio-ffmpeg`: traitement image/vidéo.

Le détail complet (incluant utilitaires et sous-dépendances) est dans les exports section 1.

## 5) Comment regénérer cette liste

Depuis la racine du dépôt:

```bash
mkdir -p docs/packages
conda run -n robocasa_telecom_ci conda list --export > docs/packages/conda_list_export_YYYY-MM-DD.txt
conda run -n robocasa_telecom_ci pip freeze > docs/packages/pip_freeze_YYYY-MM-DD.txt
```

Remarque: si vous utilisez un autre nom d'environnement (`robocasa_telecom`), remplacez `robocasa_telecom_ci` dans les commandes.
