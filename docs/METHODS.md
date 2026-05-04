# Méthodes utilisées

## 1) Cadre expérimental

- Domaine: apprentissage par renforcement pour manipulation robotique en simulation (RoboCasa).
- Tâche de référence: `OpenCabinet` (ou alias pédagogiques `OpenSingleDoor`, `OpenDoor`).
- Robot: `PandaOmron`.
- Objectif d'agent: maximiser la récompense cumulée et le taux de succès.

## 2) Méthode RL retenue

### Algorithme principal: SAC (Stable-Baselines3)

Raisons:
- off-policy et plus sample-efficient sur contrôle continu,
- bonne adéquation avec la manipulation continue RoboCasa,
- intégration stable dans SB3 avec replay buffer et reprise par checkpoints.

Baseline comparative:
- PPO est conservé comme baseline on-policy via `configs/train/open_single_door_ppo_baseline.yaml`.

Hyperparamètres configurés dans `configs/train/*.yaml`:
- SAC: `learning_rate`, `buffer_size`, `learning_starts`, `batch_size`, `tau`, `gamma`, `train_freq`, `gradient_steps`, `ent_coef`.
- PPO: `learning_rate`, `n_steps`, `batch_size`, `gamma`, `gae_lambda`, `clip_range`, `ent_coef`, `vf_coef`, `n_epochs`.

## 3) Représentation observation/action

### Action space
- directement issu de `raw_env.action_spec`, converti en `gymnasium.spaces.Box(float32)`.

### Observation space
- cas 1: wrapper gym robosuite stable (`GymWrapper`) -> observation native.
- cas 2 (fallback nominal sur cette stack): observation dict flattenée en vecteur 1D via `gymnasium.spaces.utils.flatten`.

Le fallback garantit la compatibilité SB3 quand le `GymWrapper` renvoie une shape d'observation instable entre resets.

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
5. construction PPO ou SAC,
6. entraînement avec checkpoints périodiques et validation,
7. sauvegarde modèle final + métadonnées de reprise,
8. mini-évaluation post-train,
9. export `training_curve.csv`, `validation_curve.csv` et `train_summary.json`.

## 6) Pipeline évaluation

Implémentation: `robocasa_telecom/rl/evaluate.py`.

Étapes:
1. chargement config + checkpoint,
2. rollouts sur `num_episodes`,
3. agrégation `return_mean`, `return_std`, `success_rate`, `episode_length_*`, `action_magnitude_*`, `door_angle_final_*` si disponible,
4. export JSON horodaté.

## 7) Reproductibilité

Mécanismes appliqués:
- seed explicite pour train/eval,
- commits figés robocasa/robosuite dans le script de setup,
- reprise possible depuis checkpoints périodiques + replay buffer SAC,
- auto-resume du dernier run incomplet compatible,
- sauvegarde du résumé JSON par run,
- convention de nommage incluant seed et timestamp.

## 8) Méthode cluster SLURM

- `train_array.sbatch`: parallélisation par seeds via `SLURM_ARRAY_TASK_ID`.
- `eval.sbatch`: évaluation d'un checkpoint unique avec export métriques.
- wrappers shell uniformisent local/cluster avec les mêmes interfaces.

## 9) Limites connues

- performance dépendante des assets et de la version MuJoCo/driver GPU,
- warnings non bloquants `mink`, `mimicgen` et `gym` hérités des dépendances amont,
- pas de benchmark multi-tâches à ce stade.

## 10) Extensions recommandées

- ajouter normalisation d'observation et benchmarking multi-seeds automatisé,
- ajouter comparaison image-based si la deadline le permet,
- ajouter suite de benchmarks multi-tâches RoboCasa.
