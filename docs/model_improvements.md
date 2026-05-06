# Model Improvements and Future Work

> This document identifies concrete improvements to the current project, each linked to a concept from the IA705 — Learning for Robotics course.

---

## 1. Improvements Directly Linked to Course Concepts

### 1.1 Better Reward Shaping — Ablation Study

**Current state:** The 7-component reward was designed by engineering judgment validated on a single 300k debug run.

**Improvement:** Run a systematic ablation — remove or zero out each penalty one at a time and measure the effect on `approach_frac_mean`, `val_success_rate`, and `stagnation_steps_mean`:

```yaml
# Example ablation configs (create one YAML per ablation)
# ablation_no_stagnation.yaml
reward:
  w_stagnation: 0.0   # remove stagnation penalty

# ablation_no_oscillation.yaml
reward:
  w_oscillation: 0.0  # remove oscillation penalty

# ablation_no_gating.yaml  (modify reward.py to always enable approach)
```

**Course link:** Reward shaping — understanding which components actually prevent reward hacking, vs. which are redundant or even harmful.

**Expected outcome:** identify the 2–3 components that matter most; simplify the reward function.

---

### 1.2 Curriculum Learning

**Current state:** The task is presented at full difficulty from the first episode — the door starts fully closed, the robot is placed at a fixed position facing the cabinet.

**Improvement:** Implement a curriculum by varying the initial door angle:

```python
# Stage 1: door pre-opened at 50%
initial_theta = 0.5

# Stage 2: door pre-opened at 20%
initial_theta = 0.2

# Stage 3: full task, door fully closed
initial_theta = 0.0
```

This can be implemented in `envs/factory.py` by passing an `initial_door_angle` parameter to the RoboCasa environment initialisation.

**Course link:** Curriculum learning — progressive task difficulty reduces the exploration challenge. The agent accumulates success reward earlier, making credit assignment tractable.

**Expected outcome:** significantly faster convergence (first success in < 100k steps vs. several hundred thousand), higher final success rate.

---

### 1.3 Imitation Learning — BC Pre-training + SAC Fine-tuning

**Current state:** The agent starts from a randomly-initialised policy. The first useful actions (handle contact) are discovered by chance.

**Improvement:**

1. **Collect demonstrations** using RoboCasa's scripted policies or human teleoperation (available in `robocasa/demos/`).
2. **Behavioral Cloning (BC) pre-training** — train the actor network on `(state, action)` pairs from demonstrations with supervised cross-entropy / MSE loss.
3. **SAC fine-tuning** — initialise SAC's actor with the BC-trained weights; use RL to correct distribution shift and achieve higher performance than BC alone.

```python
# Conceptual BC pre-training
from robocasa.utils import collect_demos
demos = collect_demos("OpenCabinet", n=50)
bc_train(actor_network, demos, epochs=100)
model = SAC.load_pretrained(actor_network, ...)
model.learn(total_timesteps=3_000_000)
```

**Course link:** Imitation learning — Behavioral Cloning, covariate shift, and the benefit of combining demonstration pre-training with RL fine-tuning to avoid random exploration.

**Expected outcome:** the agent achieves non-zero success rate from the first evaluation (~step 25k), dramatically reducing wasted compute on random exploration.

---

### 1.4 More Controlled Exploration

**Current state:** SAC uses automatic entropy tuning (`ent_coef=auto`) which adapts the exploration level automatically. PPO uses a fixed entropy bonus (`ent_coef=0.01`).

**Improvement options:**

- **Normalise observations** using `VecNormalize` (SB3 built-in) — observation normalisation significantly improves SAC stability on tasks where different observation dimensions have very different scales (e.g., joint angles in radians vs. distances in meters).
- **Schedule the entropy coefficient** — start with high entropy (high exploration), decay toward lower entropy as training progresses and the policy becomes more directed.
- **Add observation noise** during training for robustness.

```yaml
# Add to train config
train:
  normalize_observations: true
  ent_coef: "auto_0.1"  # target entropy = -0.1 * action_dim
```

**Course link:** Exploration vs. exploitation — controlling the exploration schedule is a core RL design choice. The course covers entropy regularisation as a principled exploration mechanism.

---

### 1.5 Observation Normalisation and Reward Normalisation

**Current state:** Raw observations (joint angles, distances) and raw rewards are fed to the network without normalisation.

**Improvement:** Use `VecNormalize` from Stable-Baselines3:

```python
from stable_baselines3.common.vec_env import VecNormalize
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
```

This maintains a running mean and standard deviation for all observation dimensions and scales rewards to have unit variance. Known to improve SAC convergence speed and stability on manipulation tasks.

**Course link:** Policy gradient variance reduction — normalising observations reduces gradient variance; normalising rewards prevents the critic from being dominated by outlier reward values.

---

### 1.6 Comparison PPO vs. SAC vs. TD3

**Current state:** SAC (main) + PPO (baseline) are implemented. No TD3 comparison.

**Improvement:** Add TD3 (Twin Delayed DDPG) as a third algorithm:

```yaml
# configs/train/open_single_door_td3.yaml
train:
  algorithm: TD3
  learning_rate: 3.0e-4
  buffer_size: 1_000_000
  batch_size: 256
  tau: 0.005
  policy_delay: 2
  target_policy_noise: 0.2
  noise_clip: 0.5
```

TD3 is also actor-critic off-policy like SAC, but without entropy regularisation. Comparing SAC and TD3 isolates the contribution of entropy maximisation to exploration and performance.

**Course link:** Off-policy actor-critic methods — understanding the trade-offs between SAC (entropy-augmented), TD3 (deterministic policy + noise), and PPO (on-policy).

---

### 1.7 Multi-Seed Evaluation

**Current state:** All experiments use a single seed (seed=0). RL results are highly seed-dependent.

**Improvement:** Run 3–5 seeds per algorithm and report mean ± standard deviation:

```bash
for seed in 0 1 2 3 4; do
    make train-sac SEED=$seed
done

# Plot with confidence intervals
uv run python scripts/plot_training.py \
    --run outputs/OpenCabinet_SAC_seed0_*/ \
          outputs/OpenCabinet_SAC_seed1_*/ \
          outputs/OpenCabinet_SAC_seed2_*/ \
    --label "SAC s0" "SAC s1" "SAC s2" \
    --out outputs/plots/multi_seed/
```

**Course link:** Statistical robustness in RL — the course emphasises that single-seed results are anecdotes, not evidence. The standard in the RL literature is 5–10 seeds with confidence intervals.

---

### 1.8 Systematic Ablations

A full ablation study would include:

| Ablation | What it tests |
|---|---|
| No stagnation penalty (`w_stagnation=0`) | Effect of stagnation penalty on hover hacking |
| No oscillation penalty (`w_oscillation=0`) | Effect of oscillation detection |
| No approach gating (always-on approach) | Effect of gating on late-training behaviour |
| No high-watermark (use raw Δθ) | Effect of high-watermark on oscillation exploitation |
| `gradient_steps=1` vs. `gradient_steps=12` | SAC sample efficiency vs. wall-clock speed trade-off |
| `n_envs=1` vs. `n_envs=6` vs. `n_envs=12` | Effect of parallelism on training quality |

**Course link:** Ablation studies are the standard tool for understanding which components of a method contribute to performance. The course covers this in the context of reward design and algorithm analysis.

---

### 1.9 Replay Buffer Inspection (SAC)

**Current state:** The SAC replay buffer is saved as a `.pkl` file but never analysed.

**Improvement:** Periodically sample from the replay buffer to analyse:
- Distribution of rewards (are most transitions near-zero reward? This indicates wasted exploration.)
- Distribution of `door_angle` values in the buffer (are there any high-θ transitions? If not, the agent has never opened the door far enough.)
- Success rate among buffered transitions (what fraction led to the success bonus?)

This diagnostic reveals whether the replay buffer contains useful learning signal or mostly useless near-zero-reward transitions.

**Course link:** Off-policy learning — the quality of the replay buffer directly determines what the agent can learn. Understanding buffer composition is a key diagnostic tool.

---

### 1.10 Better Success Condition Definition

**Current state:** Success is defined as `θ ≥ 0.90` — the door must be 90% open at some point during the episode.

**Improvement options:**

- **Sustained success** — require `θ ≥ 0.90` for ≥ 10 consecutive steps, preventing the agent from momentarily reaching the threshold and then immediately releasing.
- **End-state success** — require `θ ≥ 0.90` at the final step, not at any intermediate step.
- **Held-out configuration success** — evaluate on cabinet types and positions not seen during training, testing genuine generalisation.

**Course link:** Evaluation protocol — the definition of "success" affects all reported metrics and conclusions. The course covers the importance of aligning the evaluation criterion with the true task objective.

---

### 1.11 Policy Evaluation on Held-Out Configurations

**Current state:** The validation split uses `seed=10000` and the test split uses `seed=20000`. However, these seeds randomise the scene appearance but not the cabinet type or robot starting position range.

**Improvement:** Define genuinely held-out evaluation sets:
- Different cabinet styles (not seen during training)
- Different distances between robot and cabinet
- Different lighting conditions

**Course link:** Generalisation in robot learning — a policy that works only for the exact training distribution is not useful. The course covers the sim-to-real gap and the importance of out-of-distribution evaluation.

---

## 2. Summary Table

| Improvement | Difficulty | Impact | Course concept |
|---|---|---|---|
| Ablation study on reward | Low | High | Reward shaping analysis |
| Curriculum learning | Medium | High | Curriculum learning |
| BC pre-training + SAC fine-tuning | Medium | Very high | Imitation learning |
| Observation normalisation | Low | Medium | Variance reduction |
| Multi-seed evaluation | Low (compute cost) | High | Statistical robustness |
| TD3 comparison | Low | Medium | Off-policy actor-critic |
| Systematic ablations | Medium | High | Scientific methodology |
| Replay buffer inspection | Low | Medium | Off-policy learning |
| Sustained success criterion | Low | Medium | Evaluation protocol |
| Held-out configuration eval | Medium | High | Generalisation |
| Navigation integration | Very high | Very high | Full manipulation+navigation |
| Sim-to-real transfer | Very high | Critical | Robot learning |

---

## 3. Roadmap Priority (Post-Deadline)

If this project were to be continued:

1. **Short term (1–2 weeks):** Multi-seed evaluation + observation normalisation — high impact, low effort.
2. **Medium term (1 month):** Ablations on reward components + curriculum learning — would significantly improve the scientific rigor of the results.
3. **Long term (1 semester):** Imitation learning integration + held-out evaluation — would transform this from a course project into a publishable contribution.
