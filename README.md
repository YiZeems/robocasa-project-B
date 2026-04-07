# RoboCasa Project B (Telecom)

Squelette exécutable pour entraîner et évaluer un agent RL sur RoboCasa, aligné avec `ARISE-Initiative/robosuite` (package Python racine + scripts opérationnels + configs + docs + tests).

## Objectif

- entraîner/évaluer une baseline PPO sur la tâche `OpenCabinet` (alias gérés: `OpenSingleDoor`, `OpenDoor`);
- exécuter en local et sur cluster SLURM GPU;
- produire des artefacts reproductibles (checkpoints, courbes, métriques, logs).

## Compatibilité OS (important)

Le projet supporte:

- Linux: recommandé (local/cluster)
- macOS: supporté pour dev/tests locaux
- Windows: supporté via Git Bash ou WSL (recommandé), pas via `cmd.exe` brut

Les scripts `scripts/setup_conda.sh` et `scripts/with_env.sh` détectent l'OS automatiquement et affichent des indications adaptées.

## Architecture attendue après setup

Après `git clone`, certains éléments sont absents du dépôt (normal, gitignore):

- `external/robosuite`, `external/robocasa` (clonés par setup)
- assets RoboCasa (`external/robocasa/robocasa/models/assets/...`)
- artefacts (`checkpoints`, `outputs`, `logs`)
- environnement conda local

Structure du dépôt:

```text
.
├── robocasa_telecom/           # package principal
│   ├── envs/                   # factory env + adapters
│   ├── rl/                     # train/eval
│   ├── tools/                  # sanity
│   └── utils/
├── scripts/
│   ├── setup_conda.sh
│   ├── with_env.sh
│   ├── run_train.sh
│   ├── run_eval.sh
│   └── slurm/
├── configs/
├── docs/
├── tests/
├── environment.yml
└── requirements-project.txt
```

## Ressources minimales

- Stockage disque:
  - minimum: `25 Go` libres
  - recommandé: `40+ Go` libres
- RAM:
  - minimum: `16 Go`
  - recommandé: `32 Go`
- Calcul:
  - minimum: CPU moderne (fonctionne mais lent)
  - recommandé: GPU NVIDIA `8 Go VRAM` ou plus

### Ce qui prend le plus de place

- `external/robocasa/robocasa/models/assets/`: principal poste (`~10-20+ Go`)
- `external/robosuite` + `external/robocasa`: code source (`~1-3 Go` cumulés)
- `checkpoints/`, `outputs/`, `logs/`: grossissent avec les runs
- env Conda `robocasa_telecom`: plusieurs Go hors dépôt

Audit rapide:

```bash
du -sh external checkpoints outputs logs 2>/dev/null
du -h -d 2 external/robocasa/robocasa/models/assets 2>/dev/null | sort -h | tail -n 20
```

## Setup complet (guide de reconstruction)

### 1) Clone

```bash
git clone git@github.com:YiZeems/robocasa-project-B.git
cd robocasa-project-B
```

### 2) Setup conda + dépendances + externals

```bash
bash scripts/setup_conda.sh
conda activate robocasa_telecom
```

Ce que fait `scripts/setup_conda.sh`:

- détecte l'OS (Linux/macOS/Windows Git Bash/WSL);
- crée/réutilise `robocasa_telecom` en Python `3.11`;
- clone + checkout les commits figés de `robosuite` et `robocasa`;
- installe en editable: `external/robosuite`, `external/robocasa`, `.` ;
- installe [requirements-project.txt](/Users/yimouzhang/Documents/Telecom_Paris/Mastere_Specialise/Intelligence_Artificielle_Multimodale/IA705_Apprentissage_pour_la_robotique/projet/robocasa-project-B/requirements-project.txt);
- télécharge/valide les assets (si `DOWNLOAD_ASSETS=1`);
- valide les imports critiques.

Si `conda activate` ne marche pas:

```bash
scripts/with_env.sh python -m robocasa_telecom.sanity --config configs/env/open_single_door.yaml --steps 20
```

### 3) Vérification post-install

```bash
python -m pip check
pytest -q
python -m robocasa_telecom.sanity --config configs/env/open_single_door.yaml --steps 20
```

## Commandes principales

Train:

```bash
python -m robocasa_telecom.train --config configs/train/open_single_door_ppo.yaml --seed 0
```

Eval (checkpoint explicite):

```bash
python -m robocasa_telecom.evaluate \
  --config configs/train/open_single_door_ppo.yaml \
  --checkpoint checkpoints/<run_id>/final_model.zip \
  --num-episodes 20 \
  --deterministic
```

Eval (dernier checkpoint automatiquement):

```bash
python -m robocasa_telecom.evaluate \
  --config configs/train/open_single_door_ppo.yaml \
  --checkpoint "$(ls -t checkpoints/*/final_model.zip | head -n 1)" \
  --num-episodes 2 \
  --deterministic
```

Visualisation:

```bash
scripts/with_env.sh python scripts/visualize_env.py --config configs/env/open_single_door.yaml --steps 200
```

## Validation fonctionnelle de bout en bout

```bash
python -m robocasa_telecom.train \
  --config configs/train/open_single_door_ppo.yaml \
  --seed 0 \
  --total-timesteps 256

python -m robocasa_telecom.evaluate \
  --config configs/train/open_single_door_ppo.yaml \
  --checkpoint "$(ls -t checkpoints/*/final_model.zip | head -n 1)" \
  --num-episodes 2 \
  --deterministic
```

Note: PPO SB3 exécute au minimum `n_steps` transitions (par défaut `2048`), même si `--total-timesteps` est plus petit.

## SLURM

```bash
sbatch scripts/slurm/train_array.sbatch
sbatch --export=ALL,CHECKPOINT_PATH=checkpoints/<run_id>/final_model.zip scripts/slurm/eval.sbatch
```

## Sorties générées

- `checkpoints/<run_id>/final_model.zip`
- `outputs/<run_id>/monitor.csv`
- `outputs/<run_id>/training_curve.csv`
- `outputs/<run_id>/train_summary.json`
- `outputs/eval/eval_YYYYmmdd_HHMMSS.json`

## Warnings connus non bloquants

- `mimicgen environments not imported`
- warning IK `mink` côté robosuite
- message `gym` legacy affiché par dépendances tierces

## Entraînement multi-algorithmes sur cluster SLURM (PPO / SAC / A2C)

### Contexte

La branche `feature/training-eval` ajoute le support de trois algorithmes (PPO, SAC, A2C) ainsi qu'un reward shaping dense basé sur la distance gripper→porte (le flag `reward_shaping` de RoboCasa étant non fonctionnel dans la version actuelle).

### Étape 1 — Cloner et se placer sur la bonne branche

```bash
git clone https://github.com/YiZeems/robocasa-project-B.git
cd robocasa-project-B
git checkout feature/training-eval
```

### Étape 2 — Installer l'environnement (une seule fois par machine)

```bash
bash scripts/setup_conda.sh
conda activate robocasa_telecom
```

Le script installe robosuite, robocasa, toutes les dépendances et télécharge les assets (~10-20 Go). Compter 30-60 min selon la connexion.

### Étape 3 — Vérifier l'installation

```bash
python -m robocasa_telecom.sanity --config configs/env/open_single_door.yaml --steps 20
```

`Sanity check passed` → tout est opérationnel.

### Étape 4 — Lancer les 3 entraînements en parallèle (500k steps)

```bash
# PPO
sbatch --export=ALL,CONFIG_PATH=configs/train/open_single_door_ppo.yaml,SEEDS_CSV=0,TOTAL_TIMESTEPS=500000 \
  scripts/slurm/train_array.sbatch

# SAC
sbatch --export=ALL,CONFIG_PATH=configs/train/open_single_door_sac.yaml,SEEDS_CSV=0,TOTAL_TIMESTEPS=500000 \
  scripts/slurm/train_array.sbatch

# A2C
sbatch --export=ALL,CONFIG_PATH=configs/train/open_single_door_a2c.yaml,SEEDS_CSV=0,TOTAL_TIMESTEPS=500000 \
  scripts/slurm/train_array.sbatch
```

### Étape 5 — Suivre la progression

```bash
squeue -u $USER
tail -f logs/slurm/robocasa_train-<JOBID>_0.out
```

### Étape 6 — Comparer les résultats

Une fois les 3 jobs terminés :

```bash
python scripts/compare_runs.py --csv results/comparison.csv
```

Pour partager les résultats via git (les `outputs/` sont exclus du `.gitignore`) :

```bash
git add -f outputs/OpenCabinet_ppo_seed0_*/train_summary.json
git add -f outputs/OpenCabinet_sac_seed0_*/train_summary.json
git add -f outputs/OpenCabinet_a2c_seed0_*/train_summary.json
git commit -m "results: 500k training results PPO/SAC/A2C"
git push origin feature/training-eval
```

## Documentation détaillée

- [Architecture détaillée](docs/ARCHITECTURE.md)
- [Méthodes RL et choix techniques](docs/METHODS.md)
- [Runbook local + cluster](docs/RUNBOOK.md)
- [Organisation équipe de 4](docs/COLLABORATION.md)
- [Packages installés/utilisés](docs/PACKAGES.md)
- [CI/CD Linux](docs/CI.md)
- [Référence fichier par fichier](docs/FILE_REFERENCE.md)
