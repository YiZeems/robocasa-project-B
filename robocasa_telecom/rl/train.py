
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from ..envs.factory import EnvConfig, load_env_config, make_env_from_config
from ..utils.io import ensure_dir, load_yaml
from ..utils.success import infer_success

                                                     
ALGO_MAP = {
    "PPO": PPO,
    "SAC": SAC,
    "A2C": A2C,
}


def _make_env_fn(env_cfg: EnvConfig, seed: int | None) -> Any:
    return make_env_from_config(env_cfg, seed=seed)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Train RL agent on RoboCasa task")
    parser.add_argument("--config", required=True, help="Path to train YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help=(
            "Total timesteps to train. "
            "When --resume-from is used, this is the number of *additional* steps "
            "on top of the checkpoint's current step count."
        ),
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help=(
            "Path to a .zip checkpoint to resume training from. "
            "The environment is re-created from --config as usual. "
            "--total-timesteps then means additional steps beyond the checkpoint's step count. "
            "Example: --resume-from checkpoints/run_id/final_model.zip --total-timesteps 500000"
        ),
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Override n_envs from YAML. Reduce if SubprocVecEnv crashes with "
            "'Could not allocate memory', BrokenPipeError, or EOFError. "
            "Recommended: try 8, then 6 on Windows with RoboCasa."
        ),
    )
    return parser.parse_args()


def _build_model(
    algorithm: str,
    train_cfg: dict[str, Any],
    env: Monitor,
    tensorboard_root: Path,
    seed: int,
) -> Any:
    algo_cls = ALGO_MAP[algorithm]
    common = dict(
        policy=train_cfg.get("policy", "MlpPolicy"),
        env=env,
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
        gamma=float(train_cfg.get("gamma", 0.99)),
        tensorboard_log=str(tensorboard_root),
        seed=seed,
        device=train_cfg.get("device", "auto"),
        verbose=1,
    )

    if algorithm in ("PPO", "A2C"):
        common.update(
            n_steps=int(train_cfg.get("n_steps", 2048)),
            gae_lambda=float(train_cfg.get("gae_lambda", 0.95)),
            ent_coef=float(train_cfg.get("ent_coef", 0.0)),
            vf_coef=float(train_cfg.get("vf_coef", 0.5)),
        )
        if algorithm == "PPO":
            common.update(
                batch_size=int(train_cfg.get("batch_size", 256)),
                clip_range=float(train_cfg.get("clip_range", 0.2)),
                n_epochs=int(train_cfg.get("n_epochs", 10)),
            )

    elif algorithm == "SAC":
        common.update(
            buffer_size=int(train_cfg.get("buffer_size", 100000)),
            batch_size=int(train_cfg.get("batch_size", 256)),
            tau=float(train_cfg.get("tau", 0.005)),
            ent_coef=train_cfg.get("ent_coef", "auto"),
            train_freq=int(train_cfg.get("train_freq", 1)),
            gradient_steps=int(train_cfg.get("gradient_steps", 1)),
            learning_starts=int(train_cfg.get("learning_starts", 1000)),
        )

    return algo_cls(**common)


def _evaluate_policy(model: Any, env: Monitor, episodes: int) -> dict[str, Any]:

    returns = []
    success_flags = []
    max_opens: list[float] = []
    final_opens: list[float] = []
    min_dists: list[float] = []
    final_dists: list[float] = []
    steps_near_10: list[int] = []
    steps_near_5: list[int] = []
    delta_positive_steps: list[int] = []                        
    delta_sums: list[float] = []                                                        
    hover_no_open: list[int] = []                                                 
    grasp_means: list[float] = []                                                

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_success = False
        max_open = 0.0
        min_dist = float("inf")
        open_now = 0.0
        prev_open = 0.0
        dist_now = float("nan")
        n_near_10 = 0
        n_near_5 = 0
        n_delta_pos = 0
        ep_delta_sum = 0.0
        n_hover_no_open = 0
        grasp_vals: list[float] = []                                   

        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_success = ep_success or infer_success(info, env)
            done = bool(terminated or truncated)

                                                                        
            if "_dbg_open" in info:
                open_now = float(info["_dbg_open"])
                max_open = max(max_open, open_now)
            elif hasattr(obs, "__len__") and len(obs) >= 4:
                open_now = float(obs[-4])
                max_open = max(max_open, open_now)

            delta_open = open_now - prev_open
            if delta_open > 1e-6:
                n_delta_pos += 1
                ep_delta_sum += delta_open
            prev_open = open_now

                                                                             
            close_to_handle = False
            if "_dbg_dist" in info:
                d = float(info["_dbg_dist"])
                if not np.isnan(d):
                    dist_now = d
                    if d < min_dist:
                        min_dist = d
                    if d < 0.10:
                        n_near_10 += 1
                        close_to_handle = True
                    if d < 0.05:
                        n_near_5 += 1
            elif hasattr(obs, "__len__") and len(obs) >= 4:
                eef_vec = np.asarray(obs[-3:], dtype=float)
                if np.linalg.norm(eef_vec) > 1e-6:
                    d = float(np.linalg.norm(eef_vec))
                    dist_now = d
                    if d < min_dist:
                        min_dist = d
                    if d < 0.10:
                        n_near_10 += 1
                        close_to_handle = True
                    if d < 0.05:
                        n_near_5 += 1

                                                                                   
            if close_to_handle and delta_open <= 1e-6:
                n_hover_no_open += 1

                                                                                  
            if close_to_handle:
                if "_dbg_r_grasp" in info:
                    grasp_vals.append(float(info["_dbg_r_grasp"]))
                elif hasattr(obs, "__len__") and len(obs) >= 6:
                                                                                    
                    pass

        returns.append(ep_return)
        success_flags.append(float(ep_success))
        max_opens.append(max_open)
        final_opens.append(open_now)
        min_dists.append(min_dist if min_dist < float("inf") else float("nan"))
        final_dists.append(dist_now if not np.isnan(dist_now) else float("nan"))
        steps_near_10.append(n_near_10)
        steps_near_5.append(n_near_5)
        delta_positive_steps.append(n_delta_pos)
        delta_sums.append(ep_delta_sum)
        hover_no_open.append(n_hover_no_open)
        grasp_means.append(float(np.mean(grasp_vals)) if grasp_vals else 0.0)

    def _mean(lst: list[float]) -> float:
        return float(np.mean(lst)) if lst else 0.0

    def _nanmean(lst: list[float]) -> float | None:
        arr = np.array(lst, dtype=float)
        valid = arr[~np.isnan(arr)]
        return float(np.mean(valid)) if len(valid) > 0 else None

    valid_dist_count = int(np.sum(~np.isnan(np.array(min_dists, dtype=float))))

    return {
        "eval_return_mean": _mean(returns),
        "eval_return_std": float(np.std(returns)) if returns else 0.0,
        "eval_success_rate": _mean(success_flags),
        "eval_max_open_mean": _mean(max_opens),
        "eval_final_open_mean": _mean(final_opens),
        "eval_min_dist_mean": _nanmean(min_dists),
        "eval_final_dist_mean": _nanmean(final_dists),
        "eval_valid_dist_count": valid_dist_count,
        "eval_steps_near_10cm_mean": _mean(steps_near_10),
        "eval_steps_near_5cm_mean": _mean(steps_near_5),
        "eval_open_delta_positive_steps_mean": _mean(delta_positive_steps),
        "eval_open_delta_sum_mean": _mean(delta_sums),
        "eval_hover_without_open_mean": _mean(hover_no_open),
        "eval_grasp_mean": _mean(grasp_means),
    }


def _export_training_curve(monitor_path: Path, out_csv: Path) -> None:

    if not monitor_path.exists():
        return

    rows: list[dict[str, float]] = []
    with monitor_path.open("r", encoding="utf-8") as f:
                                                                                  
        reader = csv.DictReader(line for line in f if not line.startswith("#"))
        for idx, row in enumerate(reader):
            rows.append(
                {
                    "episode": float(idx),
                    "reward": float(row.get("r", 0.0)),
                    "length": float(row.get("l", 0.0)),
                    "time": float(row.get("t", 0.0)),
                }
            )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "reward", "length", "time"])
        writer.writeheader()
        writer.writerows(rows)


def _resolve_run_context(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:

    env_cfg_path = cfg.get("env", {}).get("config_path", "configs/env/open_single_door.yaml")
    env_cfg = load_env_config(env_cfg_path)

    train_cfg = cfg.get("train", {})
    paths_cfg = cfg.get("paths", {})

    seed = int(args.seed if args.seed is not None else train_cfg.get("seed", 0))
    total_timesteps = int(
        args.total_timesteps
        if args.total_timesteps is not None
        else train_cfg.get("total_timesteps", 200000)
    )
    algorithm = train_cfg.get("algorithm", "PPO").upper()
    if algorithm not in ALGO_MAP:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Supported: {list(ALGO_MAP)}")

    run_id = f"{env_cfg.task}_{algorithm.lower()}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return {
        "env_cfg": env_cfg,
        "train_cfg": train_cfg,
        "paths_cfg": paths_cfg,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "algorithm": algorithm,
        "run_id": run_id,
    }


def _make_envs(
    env_cfg: Any,
    seed: int,
    n_envs: int,
    run_output_dir: Path,
    vec_env_cls: str = "dummy",
) -> tuple[Any, Any, Path]:
    if n_envs > 1:
        monitor_dir = ensure_dir(run_output_dir / "monitors")

        if vec_env_cls == "subproc":
            from functools import partial

                                                                                        
            if env_cfg.obs_keys is None:
                _probe = make_env_from_config(env_cfg, seed=None)
                _adapter = _probe._env                                                
                env_cfg.obs_keys = _adapter._obs_keys
                env_cfg.obs_shapes = {
                    k: tuple(v.shape) for k, v in _adapter._dict_space.spaces.items()
                }
                _probe.close()
                print(f"[train] canonical obs (probe): {len(env_cfg.obs_keys)} keys, "
                      f"base_size={sum(s[0] for s in env_cfg.obs_shapes.values())}")
            else:
                print(f"[train] canonical obs (pre-loaded): {len(env_cfg.obs_keys)} keys, "
                      f"base_size={sum(s[0] for s in env_cfg.obs_shapes.values())}")

                                                                           
            fns = [partial(_make_env_fn, env_cfg, None) for _ in range(n_envs)]
            print(f"[train] Starting {n_envs} SubprocVecEnv workers "
                  f"(if this hangs or crashes with BrokenPipeError/EOFError/"
                  f"'Could not allocate memory', retry with --n-envs 6)")
            try:
                train_env = SubprocVecEnv(fns)
            except Exception as exc:
                raise RuntimeError(
                    f"\n[train] SubprocVecEnv failed with n_envs={n_envs}: {exc}\n"
                    "  Common causes on Windows + RoboCasa:\n"
                    "    • 'Could not allocate memory' — too many MuJoCo scenes init at once\n"
                    "    • BrokenPipeError / EOFError — worker process killed during init\n"
                    "  Fix: reduce n_envs via --n-envs or use the 6-env YAML:\n"
                    "    python -m robocasa_telecom.rl.train "
                    "--config configs/train/open_single_door_ppo_parallel_6env.yaml ...\n"
                    "    or add --n-envs 6 to your current command"
                ) from exc
                                                                       
                                                                            
            vec_monitor_path = monitor_dir / "vec_monitor"
            train_env = VecMonitor(train_env, str(vec_monitor_path))
            monitor_path = Path(str(vec_monitor_path) + ".monitor.csv")
        else:
                                                                 
            def _make_fn():
                return make_env_from_config(env_cfg, seed=None)

            train_env = make_vec_env(
                _make_fn, n_envs=n_envs, monitor_dir=str(monitor_dir), seed=seed
            )
            monitor_path = run_output_dir / "monitor.csv"

        eval_env = Monitor(make_env_from_config(env_cfg, seed=seed))
        return train_env, eval_env, monitor_path

    monitor_path = run_output_dir / "monitor.csv"
    single_env = Monitor(make_env_from_config(env_cfg, seed=seed), filename=str(monitor_path))
    return single_env, single_env, monitor_path


def main() -> None:

    args = parse_args()
    cfg = load_yaml(args.config)
    context = _resolve_run_context(cfg, args)

    env_cfg = context["env_cfg"]
    train_cfg = context["train_cfg"]
    paths_cfg = context["paths_cfg"]
    seed = int(context["seed"])
    total_timesteps = int(context["total_timesteps"])
    algorithm = str(context["algorithm"])
    device = train_cfg.get("device", "auto")

    output_root = ensure_dir(paths_cfg.get("output_root", "outputs"))
    checkpoint_root = ensure_dir(paths_cfg.get("checkpoint_root", "checkpoints"))
    tensorboard_root = ensure_dir(paths_cfg.get("tensorboard_root", "logs/tensorboard"))

    n_envs = int(args.n_envs if args.n_envs is not None else train_cfg.get("n_envs", 1))
    vec_env_cls = str(train_cfg.get("vec_env_cls", "dummy")).lower()
    n_steps = int(train_cfg.get("n_steps", 2048))
    rollout_size = n_steps * n_envs

    if vec_env_cls == "subproc":
                                                                         
                                                                         
        import multiprocessing
        multiprocessing.set_start_method("spawn", force=False)

                                                                            
    try:
        import torch as _torch
        _cuda_available = _torch.cuda.is_available()
        _torch_version = _torch.__version__
        _torch_cuda = getattr(_torch.version, "cuda", "N/A") or "N/A"
        _gpu_name = _torch.cuda.get_device_name(0) if _cuda_available else "N/A"
    except ImportError:
        _cuda_available = False
        _torch_version = "N/A"
        _torch_cuda = "N/A"
        _gpu_name = "N/A"

    if device == "cuda" and not _cuda_available:
        print(
            "[train] WARNING: device=cuda requested but torch.cuda.is_available()=False. "
            "Falling back to CPU. Install a CUDA-enabled PyTorch build to use GPU."
        )
        device = "cpu"

    _effective_device = "cuda" if (device in ("auto", "cuda") and _cuda_available) else "cpu"
    batch_size = int(train_cfg.get("batch_size", 256))

                            
    print(
        f"[train] ── Device ──────────────────────────────────────────────\n"
        f"[train]   requested={device!r}  effective={_effective_device!r}\n"
        f"[train]   torch={_torch_version}  torch.cuda={_torch_cuda}\n"
        f"[train]   cuda_available={_cuda_available}  gpu={_gpu_name!r}\n"
        f"[train]   note: MuJoCo physics + SubprocVecEnv stay CPU-bound;\n"
        f"[train]         GPU accelerates PPO policy/value network updates only.\n"
        f"[train] ── Run config ──────────────────────────────────────────\n"
        f"[train]   algorithm={algorithm}  seed={seed}\n"
        f"[train]   vec_env_cls={vec_env_cls}  n_envs={n_envs}\n"
        f"[train]   n_steps={n_steps}  batch_size={batch_size}  rollout_size={rollout_size}\n"
        f"[train]   total_timesteps={total_timesteps}\n"
        f"[train] ────────────────────────────────────────────────────────"
    )

    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

        algo_cls = ALGO_MAP[algorithm]
                                                                            
        _probe = algo_cls.load(str(resume_path), device=device)
        start_steps = _probe.num_timesteps
        del _probe

                                                                                
        _obs_config_path = resume_path.parent / "obs_config.json"
        if _obs_config_path.exists():
            with _obs_config_path.open(encoding="utf-8") as _f:
                _obs_cfg = json.load(_f)
            env_cfg.obs_keys = tuple(_obs_cfg["obs_keys"])
            env_cfg.obs_shapes = {k: tuple(v) for k, v in _obs_cfg["obs_shapes"].items()}
            _base_size = sum(int(np.prod(s)) for s in env_cfg.obs_shapes.values())
            print(f"[train] Resume: loaded obs_config ({len(env_cfg.obs_keys)} keys, "
                  f"base_size={_base_size}) from {_obs_config_path}")
        else:
            print(
                f"[train] WARNING: no obs_config.json found at {_obs_config_path}. "
                "The env probe may produce a different obs_size than the original run, "
                "causing 'Observation spaces do not match'. "
                "Re-train from scratch to generate obs_config.json automatically."
            )

        end_steps = start_steps + total_timesteps
        run_id = (
            f"{env_cfg.task}_{algorithm.lower()}_seed{seed}"
            f"_resume_{start_steps // 1000}k_to_{end_steps // 1000}k"
            f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        run_output_dir = ensure_dir(output_root / run_id)
        run_checkpoint_dir = ensure_dir(checkpoint_root / run_id)

        train_env, eval_env, monitor_path = _make_envs(
            env_cfg, seed, n_envs, run_output_dir, vec_env_cls
        )
                                                                             
                                                                 
        model = algo_cls.load(str(resume_path), env=train_env, device=device)
        model.tensorboard_log = str(tensorboard_root)
        reset_num_timesteps = False
    else:
        run_id = str(context["run_id"])
        run_output_dir = ensure_dir(output_root / run_id)
        run_checkpoint_dir = ensure_dir(checkpoint_root / run_id)

        train_env, eval_env, monitor_path = _make_envs(
            env_cfg, seed, n_envs, run_output_dir, vec_env_cls
        )
        model = _build_model(algorithm, train_cfg, train_env, tensorboard_root, seed)
        reset_num_timesteps = True

                                                                                       
    if env_cfg.obs_keys is not None and env_cfg.obs_shapes is not None:
        obs_cfg_data = {
            "obs_keys": list(env_cfg.obs_keys),
            "obs_shapes": {k: list(v) for k, v in env_cfg.obs_shapes.items()},
        }
        with (run_checkpoint_dir / "obs_config.json").open("w", encoding="utf-8") as f:
            json.dump(obs_cfg_data, f, indent=2)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, int(train_cfg.get("save_freq_steps", 50000))),
        save_path=str(run_checkpoint_dir),
        name_prefix=algorithm.lower(),
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb],
        reset_num_timesteps=reset_num_timesteps,
        tb_log_name=run_id,
    )
    final_model_path = run_checkpoint_dir / "final_model"
    model.save(str(final_model_path))

    eval_metrics = _evaluate_policy(
        model,
        eval_env,
        episodes=int(train_cfg.get("eval_episodes", 20)),
    )

    _export_training_curve(monitor_path, run_output_dir / "training_curve.csv")

    summary: dict[str, Any] = {
        "run_id": run_id,
        "algorithm": algorithm,
        "task": env_cfg.task,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "final_model": f"{final_model_path}.zip",
        **eval_metrics,
    }
    if args.resume_from:
        summary["resumed_from"] = str(args.resume_from)
        summary["start_timesteps"] = start_steps
        summary["end_timesteps"] = end_steps

    with (run_output_dir / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

                                                                       
    with (run_output_dir / "resolved_train_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    train_env.close()
    if eval_env is not train_env:
        eval_env.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
