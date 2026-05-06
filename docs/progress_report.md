# Progress Report — RoboCasa Door Opening

**Module** : IA705 — Apprentissage pour la robotique, Telecom Paris  
**Date** : 5 mai 2026  
**Statut** : entraînement SAC debug terminé · SAC principal en cours · PPO à lancer

---

## 1. Summary of Work Completed

### Environment and infrastructure

- Custom `RawRoboCasaAdapter` implemented in `envs/factory.py` — wraps RoboCasa's `OpenCabinet` task into a stable Gymnasium interface with fixed observation shape across resets.
- Anti-hacking reward shaping implemented in `envs/reward.py` — 7 components with high-watermark progress, oscillation detection, stagnation penalty, approach gating, and success dominance.
- Training pipeline in `rl/train.py` — algorithm-agnostic (SAC and PPO), 12 SubprocVecEnv workers, periodic checkpoint saving, ValidationCallback with early stopping.
- Two-pass video evaluation in `rl/eval_video.py` — scoring pass (no render) followed by reproduction pass with the same seed.
- MLflow tracking integrated at every validation step and at the end of training.

### Experiments run

| Run | Status | Steps | Val. Success | Notes |
|---|---|---:|---:|---|
| SAC debug (seed=0, 300k) | ✅ Completed | 300k | [To fill] | Reward shaping validated |
| SAC principal (seed=0, 3M) | ⏳ In progress | — | — | Main reference run |
| SAC tuned (seed=0, 2M) | ⏳ Planned | — | — | lr=1e-4, ent=auto_0.2 |
| PPO baseline (seed=0, 5M) | ⏳ Planned | — | — | Comparative baseline |

### Issues identified and resolved

| Issue | Root cause | Fix |
|---|---|---|
| `VecMonitor.reset(seed=n)` crash | Final eval used VecEnv API | Detect VecEnv; use `env.seed(n); env.reset()` |
| Auto-resume crash loop | Run detected incomplete after crash | Check `final_model.json` as fallback in `_run_is_complete()` |
| Hover hacking (approach_frac > 0.5) | Naive reward dominated by approach | High-watermark progress + approach gating |
| Video impossible during training | SubprocVecEnv workers can't render | Two-pass post-training eval_video.py |
| mlruns tracked in Git | Committed before .gitignore entry | `git rm -r --cached mlruns/` |

---

## 2. Methodological Choices Made So Far

### Why we started with small debug runs

The 300k SAC debug run was launched before the 3M run for three reasons:

1. **Reward validation** — with 12 workers and 300k steps, the training reaches ~12 evaluation episodes of 25k steps each. This is enough to detect hover hacking (`approach_frac > 0.5`) and oscillation (`sign_changes_mean`) before committing ~19 hours of compute to the main run.
2. **Infrastructure validation** — the debug run confirmed that SubprocVecEnv workers don't exhaust RAM (~10 GB total), that MLflow logging fires correctly, and that the ValidationCallback saves `best_model.zip` at the right moment.
3. **Cost management** — aborting a 300k run costs minutes; aborting a 3M run costs hours. Starting small is a sound engineering practice for RL experiments where reward design is still being refined.

The course emphasises that RL algorithms are sensitive to reward design and hyperparameters. Debug runs are the standard way to catch design errors before spending compute on long experiments.

### Why we use 12 parallel workers

12 workers (SubprocVecEnv) were chosen for the following reasons:

- **SAC with replay buffer benefits from diversity** — each worker explores with a different random seed, producing uncorrelated experiences in the replay buffer, which is known to stabilise SAC training.
- **gradient_steps=12 is compute-heavy** — SAC performs 12 gradient updates per environment step, meaning the bottleneck is GPU computation (MPS/CUDA), not data collection. 12 workers ensure the GPU is never waiting for data.
- **Hardware fit** — on a 12-core machine with 36 GB RAM, 12 workers saturate CPU parallelism without exceeding memory (measured: ~10 GB total, 36 GB available).

The limitation: `SubprocVecEnv` makes video generation during training impossible, which motivated the two-pass approach in `eval_video.py`.

### Why we inspect videos

Numerical metrics can be gamed. A robot with `return_mean = 320` could be:

- Successfully opening the door (good), or
- Hovering 5 cm from the handle for 490 steps (reward hacking).

Both produce similar returns but radically different `success_rate` values. The video provides **qualitative ground truth** that cannot be faked. If the video shows the door opening, the agent has genuinely solved the task. This is the `eval_video.py` rationale: score 20 episodes, pick the best by a reward-hacking-resistant criterion, re-render it.

### Why we monitor `door_angle_final`

`door_angle_final_mean` is the **physical measurement** of task completion. Unlike `return_mean` (which can be gamed), a high `door_angle_final_mean` means the door is actually open at the end of the episode. Monitoring its evolution over training reveals:

- Whether the agent is making physical progress (increasing mean angle over training steps)
- Whether there is a gap between `door_angle_max_mean` (best during episode) and `door_angle_final_mean` (end of episode) — a large gap indicates the door is opened then closed, signalling oscillation or wrong-direction behaviour.

### Why we audit the reward

The course covers reward hacking as a fundamental RL challenge. In this project, the shaped reward has 7 components. Without separate monitoring, it is impossible to know *which* component is driving the agent's behaviour. We log each component fraction in MLflow (`train/reward_hack/*`) specifically to detect:

- `approach_frac > 0.5` → hover hacking
- `stagnation_steps_mean > 100` → chronic near-handle stagnation
- `sign_changes_mean > 15` → oscillation pathology

This per-component logging was added after the first debug run showed that the agent was accumulating approach reward without opening the door at all.

### Why MLflow is used

RL experiments are notoriously hard to reproduce and compare. Two runs of the same algorithm with different seeds can produce very different results. MLflow solves this by:

- Logging all hyperparameters, config files, and software versions for every run.
- Enabling side-by-side metric comparison across SAC, SAC tuned, and PPO runs.
- Storing `best_model.zip`, `validation_curve.csv`, and videos as artifacts attached to the run that produced them.
- Providing a UI (`http://127.0.0.1:5000`) where the professor can verify the experimental methodology.

---

## 3. Current State of Metrics

> Values below from the SAC debug run (300k, seed=0). Main run values to be filled after completion.

| Metric | SAC debug (300k) | Target (3M run) |
|---|---|---|
| `val_success_rate` | [To fill after debug analysis] | > 80% |
| `val_approach_frac_mean` | [To fill] | < 0.3 |
| `val_door_angle_max_mean` | [To fill] | > 0.7 |
| `val_stagnation_steps_mean` | [To fill] | < 20 |
| `val_sign_changes_mean` | [To fill] | < 5 |

---

## 4. Next Steps

| Priority | Task | Command |
|---|---|---|
| 1 | Complete SAC principal (3M) | `make train-sac SEED=0` |
| 2 | Launch PPO baseline (5M) | `make train-ppo-baseline SEED=0` |
| 3 | Eval test split on SAC best | `make eval-test CONFIG=... CHECKPOINT=...` |
| 4 | Generate best-episode video | `make eval-video CONFIG=... CHECKPOINT=...` |
| 5 | Generate training curves | `make plot PLOT_RUNS="outputs/OpenCabinet_SAC_seed0_*/"` |
| 6 | Fill `docs/results.md` placeholders | Manual after runs |
| 7 | Finalise `docs/project_report.md` | Before 7 May 23:59 |

---

## 5. Open Questions

- Will SAC converge above 80% validation success rate within 3M steps?
- Is the PPO baseline significantly worse than SAC, or does it converge to a similar level given enough steps?
- Does the `approach_frac` decrease over the full 3M run, confirming the anti-hacking design works?
- Is there significant overfitting (gap between validation and test success rates)?
