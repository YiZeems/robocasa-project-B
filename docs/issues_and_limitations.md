# Issues and Limitations

> This document covers both **technical issues encountered and resolved** during development, and **inherent limitations** of the current project design that would require future work to address.

---

## 1. Technical Issues Encountered and Resolved

### 1.1 Hover Hacking (Reward Gaming)

**Description:** During the first debug runs, the agent learned to park the end-effector ~5 cm from the door handle and remain stationary, accumulating approach reward without ever opening the door. Observed metrics: `return_mean ≈ 200–400`, `val_success_rate = 0%`, `approach_frac > 0.7`.

**Root cause:** The approach reward component (`r_approach ∝ (1 − d/d_max)`) provided a dense, easily-exploitable signal. The sparse success bonus was too rare for the agent to discover during early exploration.

**Fix:** Multi-layer anti-hacking reward redesign:
- High-watermark progress reward — oscillation yields zero reward.
- Approach gating — approach reward disabled once θ ≥ 0.90.
- Stagnation penalty — triggered when near the handle (d < 0.12 m) for ≥ 50 steps without progress.
- Success bonus dominance — `w_success × r_success` is 100× the maximum possible approach reward per episode.

**Status:** Resolved. Monitored via `approach_frac_mean` in MLflow.

---

### 1.2 VecMonitor Seed Incompatibility

**Description:** `VecMonitor.reset()` does not accept the `seed` keyword argument introduced in Gymnasium ≥ 0.26. The final evaluation crashed with `TypeError: VecMonitor.reset() got an unexpected keyword argument 'seed'`.

**Root cause:** The final evaluation reused the training `VecEnv` (which wraps a VecMonitor), but the evaluation code was written for the Gymnasium 0.29 API.

**Fix:**
- `utils/metrics.py` — added `isinstance(env, VecEnv)` detection; uses `env.seed(n); env.reset()` for VecEnvs.
- `rl/train.py` — final evaluation now creates a fresh single-worker environment with `make_env_from_config()`.

**Status:** Resolved.

---

### 1.3 Auto-Resume Crash Loop

**Description:** A run that crashed after saving `final_model.zip` but before writing `train_summary.json` was detected as "incomplete" by the auto-resume mechanism, triggering an infinite loop of resume attempts.

**Root cause:** `_run_is_complete()` relied solely on `train_summary.json`. Its absence after a crash caused false negatives.

**Fix:** `utils/checkpoints.py` — `_run_is_complete()` now checks `final_model.json` as a fallback, comparing `num_timesteps >= target_total_timesteps`.

**Status:** Resolved.

---

### 1.4 Video Generation Impossible During Training

**Description:** Attempting to render frames from within SubprocVecEnv workers caused `EGL`/OpenGL context errors and process crashes.

**Root cause:** `SubprocVecEnv` spawns workers in separate OS processes. OpenGL rendering contexts are process-local and cannot be shared across processes.

**Fix:** Two-pass evaluation (`rl/eval_video.py`):
- Pass 1: score N episodes without rendering (fast).
- Pass 2: re-run the best episode with offscreen renderer in a dedicated single-worker environment, using the same seed for reproducibility.

**Status:** Resolved.

---

### 1.5 Observation Shape Instability

**Description:** The native RoboSuite `GymWrapper` occasionally returned observation arrays with different shapes between consecutive `reset()` calls, breaking Stable-Baselines3's buffer pre-allocation.

**Root cause:** RoboSuite's `GymWrapper` flattens observation dictionaries at construction time, but some object keys may be absent or differently shaped depending on scene randomisation.

**Fix:** Custom `RawRoboCasaAdapter` in `envs/factory.py` — uses `gymnasium.spaces.utils.flatten()` with a reference space computed at construction time, guaranteeing constant observation shape regardless of scene variation.

**Status:** Resolved. The adapter is the nominal path for all environments in this project.

---

### 1.6 mlruns Tracked in Git

**Description:** The `mlruns/` directory was accidentally committed to Git before the `.gitignore` entry was added, causing the repository to carry large binary artifacts.

**Fix:** `git rm -r --cached mlruns/` + commit. Pre-commit hook added to block future accidental commits of ignored files.

**Status:** Resolved.

---

## 2. Limitations Related to Method Choices

### 2.1 Reward Shaping May Bias Learning

Reward shaping, while necessary to guide exploration, introduces a **shaped reward bias**: the agent optimises the shaped reward rather than the true task objective. Even with the anti-hacking guards in place, it is theoretically possible for the agent to find a policy that maximises the shaped reward while achieving a lower task success rate than an agent trained on pure sparse reward (given enough exploration).

**Mitigation:** Always report `val_success_rate` as the primary metric; never report `return_mean` alone. Monitor `approach_frac_mean`, `stagnation_steps_mean`, and `sign_changes_mean` to detect shaped-reward exploitation.

**What cannot be fixed without more work:** The exact weight coefficients (`w_approach=0.05`, etc.) were set by engineering judgment and a single debug run. A proper ablation study would be needed to show they are near-optimal.

---

### 2.2 SAC Requires Significant Tuning

SAC has more hyperparameters than PPO:

- `learning_rate`, `buffer_size`, `batch_size`, `tau`, `gradient_steps`, `ent_coef`, `learning_starts`, `target_update_interval`

Each hyperparameter interacts with the others. The current configuration (`lr=3e-4, buffer_size=1M, gradient_steps=12`) was set based on best practices from the literature, but not systematically searched. A grid search or Bayesian optimisation (e.g., Optuna) over these parameters could improve performance significantly.

**Current limitation:** Only one SAC configuration (plus one "tuned" variant) is tested. The sensitivity of results to hyperparameter choices is unknown.

---

### 2.3 MuJoCo Simulation Does Not Capture the Real World

MuJoCo provides rigid-body physics but does not model:

- Friction variability between different real door handles
- Sensor noise and calibration errors
- Mechanical backlash and compliance in real joints
- Visual appearance changes between simulation and reality (sim-to-real gap)

A policy trained in this simulation would likely fail on a real PandaOmron robot without domain randomisation and additional adaptation techniques (domain randomisation, system identification, etc.).

**Impact:** Results are valid for the simulation environment only. No claims about real-robot performance should be made.

---

### 2.4 Contact Learning Is Inherently Difficult

The critical action for the door-opening task — grasping or applying force to the handle — requires **precise contact**. The handle is a small target (~5 cm diameter), and the agent must:

1. Approach it precisely enough to establish contact
2. Apply force in the correct direction (pulling, not pushing)
3. Maintain contact throughout the arc of the door

This is an **inherently sample-inefficient** sub-problem. In practice, we observe that early in training the agent rarely makes useful contact, leading to thousands of wasted episodes. Imitation learning or curriculum learning could dramatically reduce this exploration cost.

---

### 2.5 12 Workers Complicate Debugging

While 12 workers increase data throughput, they introduce practical debugging challenges:

- A crash in one worker can silently slow down training without raising an error in the main process.
- Associating a specific episode's metrics to a specific worker requires careful seed tracking.
- Memory usage is amplified 12×, limiting the ability to add additional instrumentation (e.g., per-worker reward logging) without hitting RAM limits.
- Log output from 12 workers is interleaved, making real-time debugging of a single episode trajectory impractical.

---

### 2.6 Small Debug Runs Do Not Prove Convergence

The 300k SAC debug run is sufficient to:
- Detect hover hacking
- Verify MLflow logging
- Validate memory usage

But it is **not sufficient** to demonstrate that the algorithm will converge to a high success rate. 300k steps with 12 workers = ~25k unique environment-steps per worker on average. This is far less than the ~250k steps per worker needed for SAC to converge on this type of task.

**Consequence:** conclusions about algorithm performance cannot be drawn from the debug run alone. The 3M-step main run is required.

---

### 2.7 Limited Compute and Time

| Constraint | Impact |
|---|---|
| ~19 h for SAC 3M (MPS, M-series) | Only one seed can be run before the deadline |
| ~30 h for PPO 5M | May need to abort early or reduce to 3M steps |
| Deadline: 7 May 2026 | Insufficient time for ablations, multi-seed, curriculum |
| Single workstation (36 GB RAM) | Cannot run multiple long experiments in parallel |

**Consequence:** the project presents results for a single seed (seed=0) per algorithm. This is a significant statistical limitation — RL results are highly seed-dependent. A robust comparison would require 3–5 seeds per algorithm with confidence intervals.

---

### 2.8 Single Seed — Unknown Variance

With a single seed per algorithm, it is impossible to distinguish between:

- "SAC is better than PPO on this task" (general claim)
- "This particular SAC seed happened to work better than this particular PPO seed" (coincidence)

The Wilson confidence interval computed over N=50 evaluation episodes within a single run does *not* address seed variance — it only quantifies within-run episode variance. Cross-seed variance would require running 3+ seeds per algorithm.

**Mitigation in the report:** present results as single-seed observations, acknowledge the limitation, and compare qualitative trends rather than making strong statistical claims.

---

### 2.9 No Ablations on Reward Components

The anti-hacking reward design has 7 components and 14 hyperparameters. Without ablation studies (e.g., removing each penalty one at a time and measuring the effect on `approach_frac` and `success_rate`), it is impossible to know:

- Which penalty contributes most to preventing hover hacking
- Whether some penalties are redundant
- Whether different coefficient values would perform better

This is identified as a limitation of the current experimental design due to time constraints.

---

### 2.10 Navigation Excluded — Simplified Task

By fixing the mobile base, the task is significantly simplified compared to the full PandaOmron capability. In a realistic deployment:

- The robot would need to navigate from an arbitrary starting position to the door.
- The navigation task interacts with the manipulation task (e.g., the robot may need to reposition during door opening).

The simplified fixed-base setting is appropriate for the project scope but should be acknowledged as a limitation when discussing the generality of the results.

---

## 3. Known Warnings (Non-Blocking)

The following warnings appear during environment initialisation and are **expected and harmless**:

```
WARNING: mimicgen environments not imported since mimicgen is not installed!
WARNING: mink environments not imported since mink is not installed!
UserWarning: WARN: Gym has been unmaintained since 0.26...
```

These come from optional RoboCasa dependencies and do not affect the `OpenCabinet` task or any functionality used in this project.
