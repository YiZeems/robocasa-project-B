"""Environment factory and compatibility adapters for RoboCasa + Robosuite.

This module centralizes all environment creation logic so train/eval/sanity use the same
instantiation path, with explicit fallbacks for version and wrapper differences.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .reward import AntiHackingReward, RewardConfig

# MuJoCo offscreen rendering backend: pick a sane default per platform if the
# user did not set MUJOCO_GL explicitly. Headless Linux (WSL2, Slurm) needs
# 'egl' or 'osmesa'; macOS uses 'cgl'; Windows uses 'wgl'. Override anytime
# with `export MUJOCO_GL=osmesa` etc.
if "MUJOCO_GL" not in os.environ:
    if sys.platform == "linux":
        os.environ["MUJOCO_GL"] = "egl"
    elif sys.platform == "darwin":
        os.environ["MUJOCO_GL"] = "cgl"
    elif sys.platform.startswith("win"):
        os.environ["MUJOCO_GL"] = "wgl"

try:
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.spaces import utils as space_utils
except ModuleNotFoundError:  # pragma: no cover - handled by runtime checks.
    gym = None
    spaces = None
    space_utils = None

from ..utils.io import load_yaml


@dataclass
class EnvConfig:
    """Typed environment parameters loaded from YAML."""

    task: str = "OpenCabinet"
    robots: str = "PandaOmron"
    controller: str | None = None
    horizon: int = 400
    control_freq: int = 20
    reward_shaping: bool = True
    use_object_obs: bool = True
    use_camera_obs: bool = False
    has_renderer: bool = False
    has_offscreen_renderer: bool = False
    render_camera: str = "robot0_agentview_center"
    camera_names: tuple[str, ...] = ("robot0_agentview_center",)
    camera_height: int = 128
    camera_width: int = 128
    ignore_done: bool = False
    obj_registries: tuple[str, ...] = ("objaverse",)
    use_gym_wrapper: bool = False
    reward_cfg: dict[str, Any] = field(default_factory=dict)


if gym is not None:
    _GymEnvBase = gym.Env
else:
    _GymEnvBase = object


class GymnasiumAdapter(_GymEnvBase):
    """Adapter for robosuite GymWrapper to enforce Gymnasium reset/step contract.

    Some robosuite versions return 4-tuples and others 5-tuples. This adapter normalizes
    the API and enforces truncation at the configured horizon.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, gym_env: Any, raw_env: Any, horizon: int):
        super().__init__()
        self._env = gym_env
        self.raw_env = raw_env
        self._horizon = int(horizon)
        self._episode_steps = 0

        # Forward action / observation spaces from wrapped env.
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset environment and always return `(obs, info)` as Gymnasium expects."""

        del options
        self._episode_steps = 0

        try:
            out = self._env.reset(seed=seed)
        except TypeError:
            # Older wrappers may not accept seed argument.
            out = self._env.reset()

        if isinstance(out, tuple) and len(out) == 2:
            return out
        return out, {}

    def step(self, action):
        """Normalize output tuple shape and enforce `truncated` on horizon overflow."""

        out = self._env.step(action)
        if not isinstance(out, tuple):
            raise RuntimeError("Unexpected step() output from wrapped environment")

        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            terminated = bool(terminated)
            truncated = bool(truncated)
        elif len(out) == 4:
            obs, reward, done, info = out
            terminated = bool(done)
            truncated = False
        else:
            raise RuntimeError(f"Unexpected step() tuple length: {len(out)}")

        self._episode_steps += 1
        if self._episode_steps >= self._horizon and not terminated:
            truncated = True

        return obs, float(reward), terminated, truncated, dict(info or {})

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


class RawRoboCasaAdapter(_GymEnvBase):
    """Adapter around raw RoboCasa env with observation flattening for SB3.

    Stable-Baselines3 PPO with MLP policy expects flat vector observations. RoboCasa often
    returns dict observations; this adapter selects numeric leaves and flattens them into a
    deterministic key order.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        raw_env: Any,
        horizon: int,
        reference_obs_space: Any = None,
        reward_cfg: dict[str, Any] | None = None,
    ):
        if spaces is None or space_utils is None:
            raise ModuleNotFoundError(
                "gymnasium is required to build RoboCasa adapters. "
                "Install the project dependencies from pyproject.toml in the project environment."
            )

        super().__init__()
        self.raw_env = raw_env
        self._horizon = int(horizon)
        self._episode_steps = 0
        self._reward_shaper = AntiHackingReward(
            RewardConfig.from_dict(reward_cfg or {})
        )

        first_obs = self.raw_env.reset()
        if isinstance(first_obs, tuple):
            first_obs = first_obs[0]

        low, high = self.raw_env.action_spec
        self.action_space = spaces.Box(
            low=np.asarray(low, dtype=np.float32),
            high=np.asarray(high, dtype=np.float32),
            dtype=np.float32,
        )

        if reference_obs_space is not None:
            # Lock obs space to the reference (train env) so val/test envs
            # produce the exact same flat shape regardless of scene variation.
            self.observation_space = reference_obs_space
            ref_size = int(np.prod(reference_obs_space.shape))
            self._obs_keys = ("_flat",)
            self._dict_space = spaces.Dict(
                {"_flat": spaces.Box(low=-np.inf, high=np.inf, shape=(ref_size,), dtype=np.float32)}
            )
            self._reference_size = ref_size
        else:
            self._obs_keys, self._dict_space = self._build_dict_space(first_obs)
            self.observation_space = space_utils.flatten_space(self._dict_space)
            self._reference_size = None


    @staticmethod
    def _select_obs(obs: Any) -> dict[str, np.ndarray]:
        """Extract numeric observation entries; fallback to single `obs` vector."""

        if isinstance(obs, dict):
            selected: dict[str, np.ndarray] = {}
            for key, value in obs.items():
                arr = np.asarray(value, dtype=np.float32)
                if arr.size == 0:
                    continue
                selected[str(key)] = arr
            if selected:
                return selected

        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        return {"obs": arr}

    def _build_dict_space(self, obs: Any):
        """Build Gym Dict space once from first observation."""

        selected = self._select_obs(obs)
        ordered_keys = tuple(sorted(selected.keys()))
        dict_space = spaces.Dict(
            {
                key: spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=np.asarray(selected[key]).shape,
                    dtype=np.float32,
                )
                for key in ordered_keys
            }
        )
        return ordered_keys, dict_space

    def _flatten_obs(self, obs: Any) -> np.ndarray:
        """Flatten observations while keeping a fixed key order across resets/episodes."""

        if self._reference_size is not None:
            # Reference mode: produce a fixed-size flat vector regardless of
            # the current scene's obs dict layout.
            selected = self._select_obs(obs)
            parts = [arr.reshape(-1) for arr in selected.values()]
            if parts:
                raw = np.concatenate(parts).astype(np.float32)
            else:
                raw = np.zeros(self._reference_size, dtype=np.float32)
            if raw.size < self._reference_size:
                out = np.zeros(self._reference_size, dtype=np.float32)
                out[: raw.size] = raw
                return out
            return raw[: self._reference_size]

        selected = self._select_obs(obs)
        aligned: dict[str, np.ndarray] = {}
        for key in self._obs_keys:
            expected_shape = self._dict_space[key].shape
            expected_size = int(np.prod(expected_shape, dtype=np.int64))
            value = selected.get(key)
            if value is None:
                aligned[key] = np.zeros(expected_shape, dtype=np.float32)
                continue

            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size < expected_size:
                padded = np.zeros(expected_size, dtype=np.float32)
                padded[: arr.size] = arr
                arr = padded
            elif arr.size > expected_size:
                arr = arr[:expected_size]

            aligned[key] = arr.reshape(expected_shape)

        flat = space_utils.flatten(self._dict_space, aligned)
        return np.asarray(flat, dtype=np.float32)

    def _extract_theta(self) -> float:
        """Normalised door opening [0=closed, 1=open] via RoboCasa fixture API."""
        fxtr = getattr(self.raw_env, "fxtr", None)
        if fxtr is None:
            return 0.0
        try:
            joint_names = getattr(fxtr, "door_joint_names", None)
            if not joint_names:
                return 0.0
            door_state = fxtr.get_joint_state(self.raw_env, joint_names)
            return float(max(door_state.values())) if door_state else 0.0
        except Exception:
            return 0.0

    def _extract_d_ee_handle(self, raw_obs: Any) -> float:
        """Distance EEF → door/handle (m) from observation dict."""
        if not isinstance(raw_obs, dict):
            return 1.0
        vec = raw_obs.get("door_obj_to_robot0_eef_pos")
        if vec is not None:
            return float(np.linalg.norm(vec))
        # Fallback: try any key that has "eef_pos" and "door" in its name.
        for key, val in raw_obs.items():
            if "eef_pos" in key and "door" in key:
                return float(np.linalg.norm(val))
        return 1.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset raw env and return flattened observation."""

        del options
        self._episode_steps = 0
        self._reward_shaper.reset()
        try:
            out = self.raw_env.reset(seed=seed)
        except TypeError:
            out = self.raw_env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            return self._flatten_obs(obs), info
        return self._flatten_obs(out), {}

    def step(self, action):
        """Support both old and new step tuple conventions from wrapped envs."""

        out = self.raw_env.step(action)
        if not isinstance(out, tuple):
            raise RuntimeError("Unexpected step() output from raw RoboCasa env")

        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            terminated = bool(terminated)
            truncated = bool(truncated)
        elif len(out) == 4:
            obs, reward, done, info = out
            terminated = bool(done)
            truncated = False
        else:
            raise RuntimeError(f"Unexpected step() tuple length: {len(out)}")

        self._episode_steps += 1
        done = terminated or truncated
        if self._episode_steps >= self._horizon and not terminated:
            truncated = True
            done = True

        theta = self._extract_theta()
        d_ee = self._extract_d_ee_handle(obs)
        shaped, components = self._reward_shaper.compute(
            theta=theta,
            d_ee_handle=d_ee,
            action=np.asarray(action, dtype=np.float32),
        )

        step_info = dict(info or {})
        step_info["reward_components"] = components
        if done:
            step_info["episode_reward_components"] = self._reward_shaper.episode_summary()

        return self._flatten_obs(obs), shaped, terminated, truncated, step_info

    def render(self):
        return self.raw_env.render()

    def close(self):
        self.raw_env.close()


def _obs_shape_matches_space(env: Any, obs: Any) -> bool:
    """Return whether an observation shape matches the declared observation space."""

    expected_shape = getattr(getattr(env, "observation_space", None), "shape", None)
    if expected_shape is None:
        return True
    return tuple(np.asarray(obs).shape) == tuple(expected_shape)


def _gym_wrapper_is_stable(env: Any, seed: int | None) -> bool:
    """Probe multiple resets to detect GymWrapper observation-shape drift."""

    probe_seeds = []
    base_seed = 0 if seed is None else int(seed)
    probe_seeds.append(base_seed)
    probe_seeds.append(base_seed + 1)

    seen_shapes: set[tuple[int, ...]] = set()
    for probe_seed in probe_seeds:
        obs, _ = env.reset(seed=probe_seed)
        obs_shape = tuple(np.asarray(obs).shape)
        if not _obs_shape_matches_space(env, obs):
            return False
        seen_shapes.add(obs_shape)

    return len(seen_shapes) == 1


def _resolve_controller_config(env_cfg: EnvConfig):
    """Resolve controller config with broad compatibility fallbacks."""

    from robosuite.controllers import load_composite_controller_config

    try:
        from robosuite.controllers import load_controller_config
    except ImportError:
        from robosuite.controllers import load_part_controller_config as load_controller_config

    if env_cfg.controller is None:
        try:
            return load_composite_controller_config(controller=None, robot=env_cfg.robots)
        except Exception:
            return load_controller_config(default_controller="OSC_POSE")

    try:
        return load_composite_controller_config(
            controller=env_cfg.controller,
            robot=env_cfg.robots,
        )
    except Exception:
        try:
            return load_controller_config(default_controller=env_cfg.controller)
        except Exception:
            # Last-resort fallback to default controller to avoid hard failure.
            return load_composite_controller_config(controller=None, robot=env_cfg.robots)


def load_env_config(path: str | Path) -> EnvConfig:
    """Load environment YAML and return normalized `EnvConfig`."""

    data = load_yaml(path)
    env_data = data.get("env", data)
    camera_names = tuple(env_data.get("camera_names", ["robot0_agentview_center"]))
    obj_registries = tuple(env_data.get("obj_registries", ["objaverse"]))

    controller_value = env_data.get("controller", None)
    if isinstance(controller_value, str) and controller_value.lower() in {"none", "null", ""}:
        controller_value = None

    return EnvConfig(
        task=env_data.get("task", "OpenCabinet"),
        robots=env_data.get("robots", "PandaOmron"),
        controller=controller_value,
        horizon=int(env_data.get("horizon", 400)),
        control_freq=int(env_data.get("control_freq", 20)),
        reward_shaping=bool(env_data.get("reward_shaping", True)),
        use_object_obs=bool(env_data.get("use_object_obs", True)),
        use_camera_obs=bool(env_data.get("use_camera_obs", False)),
        has_renderer=bool(env_data.get("has_renderer", False)),
        has_offscreen_renderer=bool(env_data.get("has_offscreen_renderer", False)),
        render_camera=env_data.get("render_camera", "robot0_agentview_center"),
        camera_names=camera_names,
        camera_height=int(env_data.get("camera_height", 128)),
        camera_width=int(env_data.get("camera_width", 128)),
        ignore_done=bool(env_data.get("ignore_done", False)),
        obj_registries=obj_registries,
        use_gym_wrapper=bool(env_data.get("use_gym_wrapper", False)),
        reward_cfg=dict(data.get("reward", {})),
    )


def make_env_from_config(env_cfg: EnvConfig, seed: int | None = None, reference_obs_space: Any = None):
    """Instantiate RoboCasa environment using the unified project configuration.

    Raises:
        RuntimeError: when required RoboCasa assets are missing.
    """

    import contextlib as _contextlib
    import io as _io
    import logging as _logging
    import warnings as _warnings

    # robosuite/__init__.py emits WARNING/INFO to stderr and robocasa emits a
    # mimicgen notice to stdout; sink both to a StringIO during the import.
    # Real errors surface as exceptions, not log messages.
    _sink = _io.StringIO()
    with (
        _contextlib.redirect_stderr(_sink),
        _contextlib.redirect_stdout(_sink),
        _warnings.catch_warnings(),
    ):
        _warnings.filterwarnings("ignore")
        import robocasa  # noqa: F401  # Register RoboCasa tasks in robosuite.
        import robosuite

    from robosuite.wrappers.gym_wrapper import GymWrapper

    # Redirect every handler added by robosuite to the same sink so subsequent
    # controller-config INFO messages are also suppressed.
    for _name in ("robosuite", "robocasa"):
        _lg = _logging.getLogger(_name)
        for _h in _lg.handlers:
            _h.stream = _sink
        _lg.setLevel(_logging.ERROR)
        _lg.propagate = False

    controller_cfg = _resolve_controller_config(env_cfg)
    task_name = env_cfg.task

    # Keep compatibility with naming found in course material and helper zip.
    task_aliases = {
        "OpenSingleDoor": "OpenCabinet",
        "OpenDoor": "OpenCabinet",
    }
    task_name = task_aliases.get(task_name, task_name)

    try:
        raw_env = robosuite.make(
            env_name=task_name,
            robots=env_cfg.robots,
            controller_configs=controller_cfg,
            horizon=env_cfg.horizon,
            control_freq=env_cfg.control_freq,
            reward_shaping=env_cfg.reward_shaping,
            use_object_obs=env_cfg.use_object_obs,
            use_camera_obs=env_cfg.use_camera_obs,
            has_renderer=env_cfg.has_renderer,
            has_offscreen_renderer=env_cfg.has_offscreen_renderer,
            render_camera=env_cfg.render_camera,
            camera_names=list(env_cfg.camera_names),
            camera_heights=env_cfg.camera_height,
            camera_widths=env_cfg.camera_width,
            ignore_done=env_cfg.ignore_done,
            obj_registries=env_cfg.obj_registries,
        )

        if env_cfg.use_gym_wrapper:
            try:
                # Initialize once before wrapping so robot metadata is populated.
                raw_env.reset()
                # `keys=None` mirrors the known RoboCasa workaround from the provided guide.
                gym_env = GymWrapper(raw_env, keys=None)
                env = GymnasiumAdapter(
                    gym_env=gym_env,
                    raw_env=raw_env,
                    horizon=env_cfg.horizon,
                )
                if not _gym_wrapper_is_stable(env, seed):
                    print(
                        "[make_env_from_config] GymWrapper observation shape is unstable; "
                        "falling back to RawRoboCasaAdapter."
                    )
                    env = RawRoboCasaAdapter(raw_env=raw_env, horizon=env_cfg.horizon)
            except Exception:
                # Fallback to raw adapter if GymWrapper breaks on a version combination.
                env = RawRoboCasaAdapter(raw_env=raw_env, horizon=env_cfg.horizon)
        else:
            env = RawRoboCasaAdapter(
                raw_env=raw_env,
                horizon=env_cfg.horizon,
                reference_obs_space=reference_obs_space,
                reward_cfg=env_cfg.reward_cfg or {},
            )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "RoboCasa assets are missing. Run setup with DOWNLOAD_ASSETS=1 "
            "or execute: python -m robocasa.scripts.download_kitchen_assets --type all"
        ) from exc

    if seed is not None:
        env.reset(seed=seed)
    return env
