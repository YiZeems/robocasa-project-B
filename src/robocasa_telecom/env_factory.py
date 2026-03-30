from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym

import robocasa  # noqa: F401  # Register RoboCasa tasks in robosuite.
import robosuite
from robosuite.controllers import (
    load_composite_controller_config,
    load_controller_config,
)
from robosuite.wrappers.gym_wrapper import GymWrapper

from .config_utils import load_yaml


@dataclass
class EnvConfig:
    task: str = "OpenSingleDoor"
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
    hard_reset: bool = True
    use_gym_wrapper: bool = True


class GymnasiumAdapter(gym.Env):
    """Adapt robosuite's GymWrapper outputs to Gymnasium's step/reset API."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, gym_env: Any, raw_env: Any, horizon: int):
        super().__init__()
        self._env = gym_env
        self.raw_env = raw_env
        self._horizon = int(horizon)
        self._episode_steps = 0

        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        self._episode_steps = 0

        try:
            out = self._env.reset(seed=seed)
        except TypeError:
            out = self._env.reset()

        if isinstance(out, tuple) and len(out) == 2:
            return out
        return out, {}

    def step(self, action):
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


class RawRoboCasaAdapter(gym.Env):
    """Fallback adapter if GymWrapper is disabled."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, raw_env: Any, horizon: int):
        super().__init__()
        self.raw_env = raw_env
        self._horizon = int(horizon)
        self._episode_steps = 0

        low, high = self.raw_env.action_spec
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=float)

        first_obs = self.raw_env.reset()
        if isinstance(first_obs, tuple):
            first_obs = first_obs[0]
        obs_shape = getattr(first_obs, "shape", None)
        if obs_shape is None:
            raise RuntimeError("Raw obs is not array-like; enable use_gym_wrapper=true")
        self.observation_space = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=obs_shape,
            dtype=float,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        self._episode_steps = 0
        try:
            out = self.raw_env.reset(seed=seed)
        except TypeError:
            out = self.raw_env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            return out
        return out, {}

    def step(self, action):
        obs, reward, done, info = self.raw_env.step(action)
        self._episode_steps += 1
        terminated = bool(done)
        truncated = self._episode_steps >= self._horizon and not terminated
        return obs, float(reward), terminated, truncated, dict(info or {})

    def render(self):
        return self.raw_env.render()

    def close(self):
        self.raw_env.close()


def _resolve_controller_config(env_cfg: EnvConfig):
    if env_cfg.controller is None:
        try:
            return load_composite_controller_config(controller=None, robot=env_cfg.robots)
        except Exception:
            return load_controller_config(default_controller="OSC_POSE")

    try:
        return load_controller_config(default_controller=env_cfg.controller)
    except Exception:
        return load_composite_controller_config(controller=env_cfg.controller, robot=env_cfg.robots)


def load_env_config(path: str | Path) -> EnvConfig:
    data = load_yaml(path)
    env_data = data.get("env", data)
    camera_names = tuple(env_data.get("camera_names", ["robot0_agentview_center"]))

    controller_value = env_data.get("controller", None)
    if isinstance(controller_value, str) and controller_value.lower() in {"none", "null", ""}:
        controller_value = None

    return EnvConfig(
        task=env_data.get("task", "OpenSingleDoor"),
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
        hard_reset=bool(env_data.get("hard_reset", True)),
        use_gym_wrapper=bool(env_data.get("use_gym_wrapper", True)),
    )


def make_env_from_config(env_cfg: EnvConfig, seed: int | None = None):
    controller_cfg = _resolve_controller_config(env_cfg)

    raw_env = robosuite.make(
        env_name=env_cfg.task,
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
        hard_reset=env_cfg.hard_reset,
    )

    if env_cfg.use_gym_wrapper:
        # keys=None follows RoboCasa workaround shown in the provided guide.
        gym_env = GymWrapper(raw_env, keys=None)
        env = GymnasiumAdapter(gym_env=gym_env, raw_env=raw_env, horizon=env_cfg.horizon)
    else:
        env = RawRoboCasaAdapter(raw_env=raw_env, horizon=env_cfg.horizon)

    if seed is not None:
        env.reset(seed=seed)
    return env
