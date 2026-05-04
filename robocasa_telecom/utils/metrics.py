"""Shared rollout metric helpers for train and evaluation runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

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


def summarize_rollout_episodes(
    model: Any,
    env: Any,
    episodes: int,
    seed: int = 0,
    deterministic: bool = True,
) -> dict[str, float]:
    """Roll out a policy and summarize return, success, length, and action stats."""

    episodes = max(1, int(episodes))

    returns: list[float] = []
    lengths: list[float] = []
    successes: list[float] = []
    action_magnitudes: list[float] = []
    door_angles: list[float] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        ep_length = 0
        ep_success = False
        ep_action_magnitudes: list[float] = []
        ep_door_angle: float | None = None

        while not done:
            action, _state = model.predict(obs, deterministic=deterministic)
            ep_action_magnitudes.append(action_magnitude(action))

            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_length += 1
            ep_success = ep_success or infer_success(info, env)

            door_angle = extract_scalar_metric(info, env, DOOR_ANGLE_KEYS)
            if door_angle is not None:
                ep_door_angle = door_angle

            done = bool(terminated or truncated)

        if ep_door_angle is None:
            ep_door_angle = extract_scalar_metric(None, env, DOOR_ANGLE_KEYS)

        returns.append(ep_return)
        lengths.append(float(ep_length))
        successes.append(float(ep_success))
        action_magnitudes.append(
            float(np.mean(ep_action_magnitudes)) if ep_action_magnitudes else 0.0
        )
        if ep_door_angle is not None:
            door_angles.append(float(ep_door_angle))

    metrics = {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "success_rate": float(np.mean(successes)),
        "episode_length_mean": float(np.mean(lengths)),
        "episode_length_std": float(np.std(lengths)),
        "action_magnitude_mean": float(np.mean(action_magnitudes)),
        "action_magnitude_std": float(np.std(action_magnitudes)),
        "num_episodes": float(episodes),
    }

    if door_angles:
        metrics["door_angle_final_mean"] = float(np.mean(door_angles))
        metrics["door_angle_final_std"] = float(np.std(door_angles))

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

