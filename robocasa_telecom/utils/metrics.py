"""Shared rollout metric helpers for train and evaluation runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from .success import infer_success


DOOR_ANGLE_KEYS = (
    "door_angle",
    "door_open_angle",
    "open_angle",
    "hinge_angle",
    "joint_position",
    "door_joint_position",
    "joint_position_door",
)

# Distance below which the EEF is considered "near the handle"
_D_PROX_DEFAULT = 0.15


def _to_float(value: Any) -> float | None:
    """Convert a scalar-like value into a float if possible."""

    if value is None:
        return None

    if isinstance(value, (bool, int, float, np.number)):
        return float(value)

    try:
        arr = np.asarray(value)
    except Exception:
        return None

    if arr.size != 1:
        return None

    try:
        return float(arr.reshape(-1)[0])
    except Exception:
        return None


def _iter_mapping_values(payload: Any):
    """Yield nested mapping values in a breadth-first walk."""

    queue = [payload]
    seen: set[int] = set()
    while queue:
        current = queue.pop(0)
        if current is None:
            continue

        obj_id = id(current)
        if obj_id in seen:
            continue
        seen.add(obj_id)

        if isinstance(current, Mapping):
            for value in current.values():
                yield value
                if isinstance(value, Mapping):
                    queue.append(value)
                elif isinstance(value, (list, tuple)):
                    queue.extend(item for item in value if isinstance(item, Mapping))
        elif isinstance(current, (list, tuple)):
            for item in current:
                yield item
                if isinstance(item, Mapping):
                    queue.append(item)


def _extract_from_mapping(payload: Any, keys: tuple[str, ...]) -> float | None:
    """Search nested mappings for the first scalar matching one of the keys."""

    if not isinstance(payload, Mapping):
        return None

    lowered_keys = tuple(key.lower() for key in keys)
    queue: list[Any] = [payload]
    seen: set[int] = set()

    while queue:
        current = queue.pop(0)
        if not isinstance(current, Mapping):
            continue

        obj_id = id(current)
        if obj_id in seen:
            continue
        seen.add(obj_id)

        for key, value in current.items():
            key_str = str(key).lower()
            if any(candidate in key_str for candidate in lowered_keys):
                scalar = _to_float(value)
                if scalar is not None:
                    return scalar

            if isinstance(value, Mapping):
                queue.append(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Mapping):
                        queue.append(item)

    for value in _iter_mapping_values(payload):
        scalar = _to_float(value)
        if scalar is not None:
            return scalar

    return None


def _iter_env_probes(env: Any | None, max_depth: int = 10):
    """Yield the environment and plausible wrapper/raw-env handles."""

    probe = env
    seen: set[int] = set()
    depth = 0
    while probe is not None and depth < max_depth:
        obj_id = id(probe)
        if obj_id in seen:
            break
        seen.add(obj_id)

        yield probe

        raw_env = getattr(probe, "raw_env", None)
        if raw_env is not None and id(raw_env) not in seen:
            yield raw_env

        unwrapped = getattr(probe, "unwrapped", None)
        if unwrapped is not None and id(unwrapped) not in seen:
            yield unwrapped

        probe = getattr(probe, "env", None)
        depth += 1


def extract_scalar_metric(
    info: Mapping[str, Any] | None,
    env: Any | None,
    keys: tuple[str, ...],
) -> float | None:
    """Extract a scalar metric from `info` or an env wrapper chain."""

    if info is not None:
        metric = _extract_from_mapping(info, keys)
        if metric is not None:
            return metric

    lowered_keys = tuple(key.lower() for key in keys)
    for probe in _iter_env_probes(env):
        for attr_name in dir(probe):
            attr_name_l = attr_name.lower()
            if not any(candidate in attr_name_l for candidate in lowered_keys):
                continue
            try:
                value = getattr(probe, attr_name)
            except Exception:
                continue
            scalar = _to_float(value)
            if scalar is not None:
                return scalar

    return None


def action_magnitude(action: Any) -> float:
    """Compute a stable action magnitude for logging."""

    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.linalg.norm(arr))


def success_confidence_interval(
    successes: list[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score interval for a binary success rate.

    More accurate than normal approximation for small n or extreme rates.
    Returns (lower, upper) bounds.
    """
    n = len(successes)
    if n == 0:
        return 0.0, 0.0
    p = float(np.mean(successes))
    # z for 95% CI ≈ 1.96
    z = 1.96 if confidence == 0.95 else float(np.abs(np.percentile(
        np.random.standard_normal(100_000), 100 * (1 - (1 - confidence) / 2)
    )))
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(max(0.0, centre - margin)), float(min(1.0, centre + margin))


def summarize_rollout_episodes(
    model: Any,
    env: Any,
    episodes: int,
    seed: int = 0,
    deterministic: bool = True,
    d_prox: float = _D_PROX_DEFAULT,
) -> dict[str, float]:
    """Roll out a policy and compute comprehensive anti-hacking metrics.

    Metrics returned:
    - return_mean / return_std / return_median
    - success_rate + 95% CI (lower/upper)
    - episode_length_mean / _std
    - action_magnitude_mean / _std
    - action_smoothness_mean  (mean ||a_t - a_{t-1}||, measures jerkiness)
    - door_angle_final_mean / _std
    - door_angle_max_mean     (high watermark per episode)
    - door_angle_sign_changes_mean  (oscillation proxy)
    - stagnation_steps_mean   (steps near handle without door progress)
    - approach_frac_mean      (approach reward / total reward, hover-hack signal)
    - reward_without_success  (mean return of failed episodes)
    """

    episodes = max(1, int(episodes))

    returns: list[float] = []
    lengths: list[float] = []
    successes: list[float] = []
    action_mags: list[float] = []
    action_smoothnesses: list[float] = []
    door_angles_final: list[float] = []
    door_angles_max: list[float] = []
    sign_changes_per_ep: list[float] = []
    stagnation_steps_per_ep: list[float] = []
    approach_fracs: list[float] = []
    returns_failure: list[float] = []

    for ep in range(episodes):
        if isinstance(env, VecEnv):
            env.seed(seed + ep)
            obs = env.reset()
        else:
            obs, _ = env.reset(seed=seed + ep)

        done = False
        ep_return = 0.0
        ep_length = 0
        ep_success = False

        ep_action_mags: list[float] = []
        ep_actions: list[np.ndarray] = []
        ep_door_angle_final: float | None = None
        ep_door_angle_max: float = 0.0
        ep_sign_changes: int = 0
        ep_stagnation: int = 0
        ep_theta_best: float = 0.0

        ep_approach_total: float = 0.0
        ep_reward_total: float = 0.0

        prev_theta: float | None = None
        prev_delta_sign: int = 0

        while not done:
            action, _state = model.predict(obs, deterministic=deterministic)
            action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
            ep_action_mags.append(float(np.linalg.norm(action_arr)))
            ep_actions.append(action_arr.copy())

            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_length += 1
            ep_success = ep_success or infer_success(info, env)

            # Door angle tracking
            theta = extract_scalar_metric(info, env, DOOR_ANGLE_KEYS)
            if theta is not None:
                ep_door_angle_final = theta
                ep_door_angle_max = max(ep_door_angle_max, theta)

                # High-watermark progress tracking for stagnation
                if theta > ep_theta_best + 1e-4:
                    ep_theta_best = theta
                else:
                    comp = info.get("reward_components", {}) if isinstance(info, dict) else {}
                    d_ee = comp.get("d_ee_handle", 1.0)
                    if d_ee < d_prox:
                        ep_stagnation += 1

                # Oscillation: count sign changes in Δθ
                if prev_theta is not None:
                    delta = theta - prev_theta
                    cur_sign = 1 if delta > 1e-4 else (-1 if delta < -1e-4 else 0)
                    if cur_sign != 0 and prev_delta_sign != 0 and cur_sign != prev_delta_sign:
                        ep_sign_changes += 1
                    if cur_sign != 0:
                        prev_delta_sign = cur_sign
                prev_theta = theta

            # Reward component tracking from info
            comp = info.get("reward_components", {}) if isinstance(info, dict) else {}
            ep_approach_total += abs(comp.get("approach", 0.0))
            ep_reward_total += abs(float(reward))

            done = bool(terminated or truncated)

        if ep_door_angle_final is None:
            ep_door_angle_final = extract_scalar_metric(None, env, DOOR_ANGLE_KEYS)

        # Action smoothness: mean ||a_t - a_{t-1}||
        if len(ep_actions) >= 2:
            diffs = [
                float(np.linalg.norm(ep_actions[i] - ep_actions[i - 1]))
                for i in range(1, len(ep_actions))
            ]
            smoothness = float(np.mean(diffs))
        else:
            smoothness = 0.0

        # Approach fraction (hover-hack signal)
        approach_frac = (
            ep_approach_total / max(ep_reward_total, 1e-8)
            if ep_reward_total > 1e-8 else 0.0
        )

        returns.append(ep_return)
        lengths.append(float(ep_length))
        successes.append(float(ep_success))
        action_mags.append(float(np.mean(ep_action_mags)) if ep_action_mags else 0.0)
        action_smoothnesses.append(smoothness)
        if ep_door_angle_final is not None:
            door_angles_final.append(float(ep_door_angle_final))
        door_angles_max.append(ep_door_angle_max)
        sign_changes_per_ep.append(float(ep_sign_changes))
        stagnation_steps_per_ep.append(float(ep_stagnation))
        approach_fracs.append(approach_frac)
        if not ep_success:
            returns_failure.append(ep_return)

    ci_lo, ci_hi = success_confidence_interval(successes)

    metrics: dict[str, float] = {
        # Core performance
        "return_mean":   float(np.mean(returns)),
        "return_std":    float(np.std(returns)),
        "return_median": float(np.median(returns)),
        "success_rate":  float(np.mean(successes)),
        "success_ci_lo": ci_lo,
        "success_ci_hi": ci_hi,
        "failure_rate":  1.0 - float(np.mean(successes)),
        # Episode length
        "episode_length_mean": float(np.mean(lengths)),
        "episode_length_std":  float(np.std(lengths)),
        # Actions
        "action_magnitude_mean": float(np.mean(action_mags)),
        "action_magnitude_std":  float(np.std(action_mags)),
        "action_smoothness_mean": float(np.mean(action_smoothnesses)),
        # Anti-hacking
        "door_angle_max_mean":       float(np.mean(door_angles_max)),
        "sign_changes_mean":         float(np.mean(sign_changes_per_ep)),
        "stagnation_steps_mean":     float(np.mean(stagnation_steps_per_ep)),
        "approach_frac_mean":        float(np.mean(approach_fracs)),
        "reward_without_success":    float(np.mean(returns_failure)) if returns_failure else float(np.mean(returns)),
        # Meta
        "num_episodes": float(episodes),
    }

    if door_angles_final:
        metrics["door_angle_final_mean"] = float(np.mean(door_angles_final))
        metrics["door_angle_final_std"]  = float(np.std(door_angles_final))

    return metrics


def prefixed_metrics(
    metrics: Mapping[str, Any],
    prefix: str,
) -> dict[str, float]:
    """Return a filtered dict with a prefix applied to numeric metrics."""

    out: dict[str, float] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        scalar = _to_float(value)
        if scalar is None:
            continue
        out[f"{prefix}{key}"] = scalar
    return out
