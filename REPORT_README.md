# RoboCasa — Project B: OpenCabinet with PandaOmron

**Telecom Paris — Reinforcement Learning Project, Spring 2026**

Training a PandaOmron robot to open kitchen cabinet doors using model-free RL (PPO)
in the [RoboCasa](https://robocasa.ai) simulation benchmark.

---

## Submission contents

| File / folder | Description |
|---------------|-------------|
| `report_analysis.ipynb` | Report notebook (analysis + all figures) |
| `REPORT_README.md` | This file |
| `checkpoints/OpenCabinet_ppo_seed0_resume_655k_to_855k_*/` | Best checkpoint (855k, max_open=0.026 rad) |
| `figures/` | All generated plots + video frames |
| `robocasa_telecom/` | Source code (training, reward, tools) |
| `configs/` | Training and environment configs |

---

## Installation

```bash
conda env create -f environment.yml
conda activate robocasa
```

Windows:
```powershell
scripts\setup\setup_windows.bat
```

---

## Evaluate the best model

```powershell
python -m robocasa_telecom.tools.record_video `
    --config configs/train/open_single_door_ppo_parallel_6env.yaml `
    --checkpoint "checkpoints/OpenCabinet_ppo_seed0_resume_655k_to_855k_20260505_230616/final_model.zip" `
    --episodes 5 `
    --cameras robot0_frontview robot0_eye_in_hand `
    --output eval_best.mp4
```

Available cameras: `robot0_frontview` (overview), `robot0_eye_in_hand` (wrist detail),
`robot0_agentview_left`, `robot0_agentview_right`, `robot0_agentview_center`.

---

## Train from scratch

```powershell
python -m robocasa_telecom.rl.train `
    --config configs/train/open_single_door_ppo_parallel_6env.yaml
```

Config: PPO, 6 parallel envs (SubprocVecEnv), 1M steps, CUDA auto-detected.

---

## Resume from checkpoint

```powershell
# --total-timesteps = ADDITIONAL steps on top of the checkpoint's current step count
python -m robocasa_telecom.rl.train `
    --config configs/train/open_single_door_ppo_parallel_6env.yaml `
    --resume-from checkpoints\<run_id>\final_model.zip `
    --total-timesteps 200000
```

---

## Reproduce figures

```bash
jupyter notebook report_analysis.ipynb
# Run all cells (Kernel > Restart & Run All)

# Export to HTML:
jupyter nbconvert --to html report_analysis.ipynb
```

---

## Key results

| Metric | Best value | At checkpoint |
|--------|-----------|---------------|
| `eval_return_mean` | 71.8 | 551k (hover artifact) / 62 at 906k |
| `eval_max_open_mean` | 0.026 rad (≈ 1.5°) | 855k |
| `eval_min_dist_mean` | 0.051 m | 957k |
| `eval_success_rate` | **0%** | All runs |

PPO was selected over SAC and A2C as the most stable and best-performing algorithm.
No algorithm achieved the strict success criterion within 1.15M timesteps.

---

## Reward function (final)

```
r = r_sparse
  + 0.1 · exp(-5 · d_handle)    # reach
  + 0.3 · θ_norm                 # door angle
  + 10.0 · max(0, Δθ)           # delta-open
  + 5.0 · closure · max(0, Δθ)  # grasp×delta (anti-flee)
```

---

## Troubleshooting

- If `SubprocVecEnv` crashes with `BrokenPipeError`: add `--n-envs 4` to the train command.
- If obs_size mismatch on resume: ensure `obs_config.json` is in the checkpoint directory.
- If CUDA not detected: install `torch` with `--index-url https://download.pytorch.org/whl/cu128`.
