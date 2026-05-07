
from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import A2C, PPO, SAC

from ..envs.factory import load_env_config, make_env_from_config
from ..envs.reward_shaping import RewardShapingWrapper, _all_body_names, _body_pos
from ..utils.io import load_yaml
from ..utils.success import infer_success

ALGO_MAP = {"PPO": PPO, "SAC": SAC, "A2C": A2C}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug reward shaping per step")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None, help=".zip checkpoint (omit with --random)")
    p.add_argument("--random", action="store_true", help="Use random policy instead of checkpoint")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _find_reward_wrapper(env: Any) -> RewardShapingWrapper | None:
    probe = env
    for _ in range(10):
        if isinstance(probe, RewardShapingWrapper):
            return probe
        raw = getattr(probe, "raw_env", None)
        if isinstance(raw, RewardShapingWrapper):
            return raw
        if raw is not None:
            probe = raw
            continue
        inner = getattr(probe, "_env", None)
        if inner is None:
            break
        probe = inner
    return None


def _verify_handle(shaped: RewardShapingWrapper) -> None:
    print("\n" + "=" * 72)
    print("  HANDLE BODY VERIFICATION")
    print("=" * 72)
    try:
        sim = shaped._env.sim
        fxtr = getattr(shaped._env, "fxtr", None)
        fxtr_name = (getattr(fxtr, "name", None) or "?") if fxtr else "?"

        all_bodies = _all_body_names(sim)
        all_handle = [b for b in all_bodies if "handle" in b.lower()]
        prefix = [b for b in all_handle if b.startswith(fxtr_name)] if fxtr_name != "?" else []

        print(f"  fixture_name      = {fxtr_name!r}")
        print(f"  all handle bodies = {all_handle}")
        if prefix:
            print(f"  prefix-filtered   = {prefix}")
        print(f"  selected handle   = {shaped._handle_body!r}")

        if shaped._handle_body:
            try:
                pos = _body_pos(sim, shaped._handle_body)
                print(f"  handle_pos        = {tuple(round(float(x), 3) for x in pos)}"
                      f"  ← EXISTS in MuJoCo ✓")
            except Exception as e:
                print(f"  handle_pos ERROR  = {e}  ← BODY MISSING in MuJoCo ✗")
        else:
            print("  ERROR: no handle body resolved — r_reach will be 0 every step ✗")
    except Exception as e:
        print(f"  Verification error: {e}")
    print("=" * 72)


def _verify_joints(shaped: RewardShapingWrapper, env: Any) -> None:
    print("\n" + "=" * 72)
    print("  JOINT STATE VERIFICATION")
    print("=" * 72)
    try:
        fxtr = getattr(shaped._env, "fxtr", None)
        raw_env = shaped._env

                                       
        try:
            djn = list(fxtr.door_joint_names) if fxtr else []
        except Exception as e:
            djn = []
            print(f"  fxtr.door_joint_names ERROR: {e}")

        print(f"  fxtr.door_joint_names = {djn}")
        print(f"  cached _joint_names   = {shaped._joint_names}")

        jnames = shaped._joint_names
        if not jnames:
            print("  ERROR: _joint_names is empty — r_open will always be 0 ✗")
        else:
                                                                               
            try:
                js_reset = fxtr.get_joint_state(raw_env, jnames)
                print(f"  get_joint_state() at reset  = {dict((k, round(v, 3)) for k, v in js_reset.items())}"
                      f"  ← expected ≈ 0.0")
            except Exception as e:
                print(f"  get_joint_state() at reset  ERROR: {e}  ← call failed ✗")
                print("=" * 72)
                return

                                                                              
            open_vals = []
            for _ in range(5):
                action = env.action_space.sample()
                _, _, term, trunc, info = env.step(action)
                open_vals.append(round(info.get("_dbg_open", float("nan")), 4))
                if term or trunc:
                    break

            print(f"  _dbg_open over 5 random steps = {open_vals}")
            if all(v == 0.0 for v in open_vals if not np.isnan(v)):
                print("  WARNING: open amount stayed at 0 — door may not move with random actions"
                      " (not necessarily a bug)")
            else:
                print("  open amount varies — joint metric is live ✓")

    except Exception as e:
        print(f"  Verification error: {e}")
    print("=" * 72 + "\n")


def _header(ep: int) -> None:
    print(f"\n{'='*72}")
    print(f"  Episode {ep}")
    print(f"{'='*72}")
    print(f"{'step':>5}  {'r_total':>8}  {'r_sparse':>8}  {'r_reach':>7}  {'r_open':>7}  "
          f"{'dist':>6}  {'open%':>6}  {'ok'}")
    print(f"{'-'*72}")


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    train_cfg = cfg.get("train", {})
    algorithm = train_cfg.get("algorithm", "PPO").upper()
    device = train_cfg.get("device", "auto")

    env_cfg_path = cfg.get("env", {}).get("config_path", "configs/env/open_single_door.yaml")
    env_cfg = load_env_config(env_cfg_path)
    env = make_env_from_config(env_cfg, seed=args.seed)

    if args.random:
        model = None
        print("[debug_reward] Running RANDOM policy")
    else:
        if args.checkpoint is None:
            raise ValueError("Provide --checkpoint or use --random")
        algo_cls = ALGO_MAP[algorithm]
        model = algo_cls.load(args.checkpoint, device=device)
        print(f"[debug_reward] Loaded {algorithm} from {args.checkpoint}")

    print(f"[debug_reward] obs_dim={env.observation_space.shape[0]}  "
          f"action_dim={env.action_space.shape[0]}")

                                                                   
    obs, _ = env.reset()
    shaped = _find_reward_wrapper(env)

    if shaped is None:
        print("[debug_reward] WARNING: could not locate RewardShapingWrapper — skipping verification")
    else:
        _verify_handle(shaped)
        _verify_joints(shaped, env)
                                                                                
        obs, _ = env.reset()

    for ep in range(1, args.episodes + 1):
        if ep > 1:
            obs, _ = env.reset()
        done = False
        step = 0
        ep_return = 0.0
        ep_success = False
        min_dist = float("inf")
        max_open = 0.0

        _header(ep)

        while not done:
            action = model.predict(obs, deterministic=True)[0] if model else env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            success = infer_success(info, env)
            ep_success = ep_success or success
            done = bool(terminated or truncated)
            step += 1

            r_sparse = info.get("_dbg_r_sparse", float("nan"))
            r_reach  = info.get("_dbg_r_reach",  float("nan"))
            r_open   = info.get("_dbg_r_open",   float("nan"))
            open_pct = info.get("_dbg_open",      float("nan"))
            dist     = info.get("_dbg_dist",      float("nan"))

            if dist < min_dist:
                min_dist = dist
            if open_pct > max_open:
                max_open = open_pct

            ok = "YES" if success else ""
            print(
                f"{step:>5}  {reward:>8.3f}  {r_sparse:>8.3f}  "
                f"{r_reach:>7.3f}  {r_open:>7.3f}  "
                f"{dist:>6.3f}  {open_pct*100:>5.1f}%  {ok}"
            )

        status = "SUCCESS" if ep_success else "fail"
        print(f"\n  Summary ep {ep}: return={ep_return:.3f}  "
              f"min_dist={min_dist:.3f}  max_open={max_open*100:.1f}%  [{status}]")

    env.close()


if __name__ == "__main__":
    main()
