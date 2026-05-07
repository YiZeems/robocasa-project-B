
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.spaces import utils as space_utils
except ModuleNotFoundError:                                                 
    gym = None
    spaces = None
    space_utils = None

from ..utils.io import load_yaml


@dataclass
class EnvConfig:

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
                                                                                  
                                                                               
    obs_keys: tuple[str, ...] | None = None
    obs_shapes: dict[str, tuple[int, ...]] | None = None


if gym is not None:
    _GymEnvBase = gym.Env
else:
    _GymEnvBase = object


class GymnasiumAdapter(_GymEnvBase):

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


class ObsAugmentWrapper(_GymEnvBase):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, env: Any):
        if spaces is None:
            raise ModuleNotFoundError("gymnasium required")
        super().__init__()
        self._env = env
        self.action_space = env.action_space
        old_shape = env.observation_space.shape
        n = old_shape[0] + 4
        self.observation_space = spaces.Box(
            low=np.full(n, -np.inf, dtype=np.float32),
            high=np.full(n, np.inf, dtype=np.float32),
            dtype=np.float32,
        )

    def _augment(self, flat_obs: np.ndarray) -> np.ndarray:
        open_amt = 0.0
        eef_to_handle = np.zeros(3, dtype=np.float32)
        try:
            shaped_env = self._env.raw_env                        
            open_amt = getattr(shaped_env, "_last_open", 0.0)
            handle_pos = getattr(shaped_env, "_last_handle_pos", None)
            eef_pos = getattr(shaped_env, "_last_eef_pos", None)
            if handle_pos is not None and eef_pos is not None:
                eef_to_handle = (handle_pos - eef_pos).astype(np.float32)
        except Exception:
            pass
        extra = np.array([open_amt, *eef_to_handle], dtype=np.float32)
        return np.concatenate([flat_obs, extra])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        out = self._env.reset(seed=seed, options=options)
        if isinstance(out, tuple):
            obs, info = out
            return self._augment(obs), info
        return self._augment(out), {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        return self._augment(obs), reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


class RawRoboCasaAdapter(_GymEnvBase):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        raw_env: Any,
        horizon: int,
        canonical_obs_keys: tuple[str, ...] | None = None,
        canonical_obs_shapes: dict[str, tuple[int, ...]] | None = None,
    ):
        if spaces is None or space_utils is None:
            raise ModuleNotFoundError(
                "gymnasium is required to build RoboCasa adapters. "
                "Install requirements-project.txt in the project environment."
            )

        super().__init__()
        self.raw_env = raw_env
        self._horizon = int(horizon)
        self._episode_steps = 0

        first_obs = self.raw_env.reset()
        if isinstance(first_obs, tuple):
            first_obs = first_obs[0]

        low, high = self.raw_env.action_spec
        self.action_space = spaces.Box(
            low=np.asarray(low, dtype=np.float32),
            high=np.asarray(high, dtype=np.float32),
            dtype=np.float32,
        )

        if canonical_obs_keys is not None and canonical_obs_shapes is not None:
                                                                            
                                                                                  
            self._obs_keys = canonical_obs_keys
            self._dict_space = spaces.Dict({
                key: spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=canonical_obs_shapes[key],
                    dtype=np.float32,
                )
                for key in canonical_obs_keys
            })
        else:
            self._obs_keys, self._dict_space = self._build_dict_space(first_obs)
        self.observation_space = space_utils.flatten_space(self._dict_space)

    @staticmethod
    def _select_obs(obs: Any) -> dict[str, np.ndarray]:

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
        selected = self._select_obs(obs)
        aligned = {}
        for key in self._obs_keys:
            canonical_shape = self._dict_space[key].shape
            canonical_size = int(np.prod(canonical_shape))
            raw = selected.get(key)
            if raw is not None:
                arr = np.asarray(raw, dtype=np.float32).flatten()
                if len(arr) != canonical_size:
                    buf = np.zeros(canonical_size, dtype=np.float32)
                    buf[: min(len(arr), canonical_size)] = arr[: min(len(arr), canonical_size)]
                    arr = buf
                aligned[key] = arr.reshape(canonical_shape)
            else:
                aligned[key] = np.zeros(canonical_shape, dtype=np.float32)
        flat = space_utils.flatten(self._dict_space, aligned)
        return np.asarray(flat, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):

        del options
        self._episode_steps = 0
        try:
            out = self.raw_env.reset(seed=seed)
        except TypeError:
            out = self.raw_env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            return self._flatten_obs(obs), info
        return self._flatten_obs(out), {}

    def step(self, action):

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
        if self._episode_steps >= self._horizon and not terminated:
            truncated = True

        return self._flatten_obs(obs), float(reward), terminated, truncated, dict(info or {})

    def render(self):
        return self.raw_env.render()

    def close(self):
        self.raw_env.close()


def _resolve_controller_config(env_cfg: EnvConfig):

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
                                                                               
            return load_composite_controller_config(controller=None, robot=env_cfg.robots)


def load_env_config(path: str | Path) -> EnvConfig:

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
    )


def make_env_from_config(env_cfg: EnvConfig, seed: int | None = None):

    import robocasa                                                       
    import robosuite
    from robosuite.wrappers.gym_wrapper import GymWrapper
    from .reward_shaping import RewardShapingWrapper

    controller_cfg = _resolve_controller_config(env_cfg)
    task_name = env_cfg.task

                                                                             
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

                                                                          
        shaped_env = RewardShapingWrapper(raw_env)

        if env_cfg.use_gym_wrapper:
            try:
                                                                                 
                shaped_env.reset()
                gym_env = GymWrapper(shaped_env, keys=None)
                adapter = GymnasiumAdapter(
                    gym_env=gym_env,
                    raw_env=raw_env,
                    horizon=env_cfg.horizon,
                )
            except Exception:
                adapter = RawRoboCasaAdapter(
                    raw_env=shaped_env,
                    horizon=env_cfg.horizon,
                    canonical_obs_keys=env_cfg.obs_keys,
                    canonical_obs_shapes=env_cfg.obs_shapes,
                )
        else:
            adapter = RawRoboCasaAdapter(
                raw_env=shaped_env,
                horizon=env_cfg.horizon,
                canonical_obs_keys=env_cfg.obs_keys,
                canonical_obs_shapes=env_cfg.obs_shapes,
            )

        env = ObsAugmentWrapper(adapter)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "RoboCasa assets are missing. Run setup with DOWNLOAD_ASSETS=1 "
            "or execute: python -m robocasa.scripts.download_kitchen_assets --type all"
        ) from exc

    if seed is not None:
        env.reset(seed=seed)
    return env
