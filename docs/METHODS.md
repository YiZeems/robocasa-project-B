# Méthodes utilisées

## 1) Cadre expérimental

- Domaine: apprentissage par renforcement pour manipulation robotique en simulation (RoboCasa).
- Tâche de référence: `OpenCabinet` (ou alias pédagogiques `OpenSingleDoor`, `OpenDoor`).
- Robot: `PandaOmron`.
- Objectif d'agent: maximiser la récompense cumulée et le taux de succès.

## 2) Méthode RL retenue

### Algorithme principal: PPO (Stable-Baselines3)

Raisons:
- baseline robuste et standard pour environnements continus,
- implémentation fiable dans SB3,
- bon compromis stabilité / simplicité pour projet académique.

Hyperparamètres configurés dans `configs/train/open_single_door_ppo.yaml`:
- `policy: MlpPolicy`
- `learning_rate`
- `n_steps`
- `batch_size`
- `gamma`
- `gae_lambda`
- `clip_range`
- `ent_coef`
- `vf_coef`
- `n_epochs`

## 3) Représentation observation/action

### Action space
- directement issu de `raw_env.action_spec`, converti en `gymnasium.spaces.Box(float32)`.

### Observation space
- cas 1: wrapper gym robosuite stable (`GymWrapper`) -> observation native.
- cas 2 (fallback): observation dict flattenée en vecteur 1D via `gymnasium.spaces.utils.flatten`.

Ce fallback garantit la compatibilité SB3 quand les sorties dict changent entre versions RoboCasa/Robosuite.

## 4) Détection du succès

Fonction: `robocasa_telecom.utils.success.infer_success`.

Stratégie hiérarchique:
1. lire `info["success"]`, `info["task_success"]` ou `info["is_success"]`,
2. sinon sonder `_check_success()` sur `raw_env`,
3. sinon remonter la chaîne de wrappers (`env.env...`) jusqu'à 10 niveaux.

Intérêt:
- rendre l'évaluation robuste malgré les variations d'API entre wrappers.

## 5) Pipeline entraînement

Implémentation: `robocasa_telecom/rl/train.py`.

Étapes:
1. chargement config train + env,
2. création de l'environnement,
3. instrumentation Monitor (`monitor.csv`),
4. construction PPO,
5. entraînement avec checkpoints périodiques,
6. sauvegarde modèle final,
7. mini-évaluation post-train,
8. export `training_curve.csv` et `train_summary.json`.

## 6) Pipeline évaluation

Implémentation: `robocasa_telecom/rl/evaluate.py`.

Étapes:
1. chargement config + checkpoint,
2. rollouts sur `num_episodes`,
3. agrégation `return_mean`, `return_std`, `success_rate`,
4. export JSON horodaté.

## 7) Reproductibilité

Mécanismes appliqués:
- seed explicite pour train/eval,
- commits figés robocasa/robosuite dans le script de setup,
- sauvegarde du résumé JSON par run,
- convention de nommage incluant seed et timestamp.

## 8) Méthode cluster SLURM

- `train_array.sbatch`: parallélisation par seeds via `SLURM_ARRAY_TASK_ID`.
- `eval.sbatch`: évaluation d'un checkpoint unique avec export métriques.
- wrappers shell uniformisent local/cluster avec les mêmes interfaces.

## 9) Limites connues

- performance dépendante des assets et de la version MuJoCo/driver GPU,
- instrumentation minimale (pas de tracking externe type Weights&Biases),
- baseline PPO unique (pas encore de comparaison multi-algo).

## 10) Extensions recommandées

- ajouter SAC pour comparaison policy-gradient vs off-policy,
- ajouter callbacks d'early stopping et normalisation d'observation,
- ajouter suite de benchmarks multi-tâches RoboCasa.
