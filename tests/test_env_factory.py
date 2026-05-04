"""Regression tests for RoboCasa environment factory fallbacks."""

from __future__ import annotations

import sys
import types

import numpy as np

import robocasa_telecom.envs.factory as factory


class _FakeRawEnv:
    def __init__(self):
        self.action_spec = (
            np.array([-1.0, -1.0], dtype=np.float32),
            np.array([1.0, 1.0], dtype=np.float32),
        )
        self._obs = {"obs": np.zeros(124, dtype=np.float32)}

    def reset(self, seed=None):
        del seed
        return self._obs

    def step(self, action):
        del action
        return self._obs, 0.0, True, {}

    def render(self):
        return None

    def close(self):
        return None


class _VariableObsRawEnv:
    def __init__(self):
        self.action_spec = (
            np.array([-1.0], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
        )
        self._reset_calls = 0

    def reset(self, seed=None):
        del seed
        self._reset_calls += 1
        if self._reset_calls == 1:
            return {"obs": np.zeros(192, dtype=np.float32)}
        return {"obs": np.zeros(206, dtype=np.float32)}

    def step(self, action):
        del action
        return {"obs": np.zeros(206, dtype=np.float32)}, 0.0, True, {}

    def render(self):
        return None

    def close(self):
        return None


class _FakeGymWrapper:
    def __init__(self, raw_env, keys=None):
        del raw_env, keys
        from gymnasium import spaces

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(96,),
            dtype=np.float32,
        )

    def reset(self, seed=None):
        del seed
        return np.zeros(110, dtype=np.float32)

    def step(self, action):
        del action
        return np.zeros(110, dtype=np.float32), 0.0, True, {}

    def close(self):
        return None


class _FakeGymWrapperDrift:
    def __init__(self, raw_env, keys=None):
        del raw_env, keys
        from gymnasium import spaces

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(96,),
            dtype=np.float32,
        )
        self._reset_calls = 0

    def reset(self, seed=None):
        del seed
        self._reset_calls += 1
        if self._reset_calls == 1:
            return np.zeros(96, dtype=np.float32), {}
        return np.zeros(110, dtype=np.float32), {}

    def step(self, action):
        del action
        return np.zeros(110, dtype=np.float32), 0.0, True, False, {}

    def close(self):
        return None


def test_make_env_falls_back_when_gym_wrapper_obs_shape_is_inconsistent(monkeypatch):
    fake_robocasa = types.ModuleType("robocasa")
    fake_robosuite = types.ModuleType("robosuite")
    fake_wrappers = types.ModuleType("robosuite.wrappers")
    fake_gym_wrapper = types.ModuleType("robosuite.wrappers.gym_wrapper")

    fake_robosuite.make = lambda **kwargs: _FakeRawEnv()
    fake_gym_wrapper.GymWrapper = _FakeGymWrapper

    monkeypatch.setattr(factory, "_resolve_controller_config", lambda env_cfg: {})
    monkeypatch.setitem(sys.modules, "robocasa", fake_robocasa)
    monkeypatch.setitem(sys.modules, "robosuite", fake_robosuite)
    monkeypatch.setitem(sys.modules, "robosuite.wrappers", fake_wrappers)
    monkeypatch.setitem(sys.modules, "robosuite.wrappers.gym_wrapper", fake_gym_wrapper)

    env = factory.make_env_from_config(factory.EnvConfig(use_gym_wrapper=True), seed=0)

    assert isinstance(env, factory.RawRoboCasaAdapter)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.shape == (124,)
    assert obs.shape == (124,)


def test_make_env_falls_back_when_gym_wrapper_drift_appears_on_second_reset(
    monkeypatch,
):
    fake_robocasa = types.ModuleType("robocasa")
    fake_robosuite = types.ModuleType("robosuite")
    fake_wrappers = types.ModuleType("robosuite.wrappers")
    fake_gym_wrapper = types.ModuleType("robosuite.wrappers.gym_wrapper")

    fake_robosuite.make = lambda **kwargs: _FakeRawEnv()
    fake_gym_wrapper.GymWrapper = _FakeGymWrapperDrift

    monkeypatch.setattr(factory, "_resolve_controller_config", lambda env_cfg: {})
    monkeypatch.setitem(sys.modules, "robocasa", fake_robocasa)
    monkeypatch.setitem(sys.modules, "robosuite", fake_robosuite)
    monkeypatch.setitem(sys.modules, "robosuite.wrappers", fake_wrappers)
    monkeypatch.setitem(sys.modules, "robosuite.wrappers.gym_wrapper", fake_gym_wrapper)

    env = factory.make_env_from_config(factory.EnvConfig(use_gym_wrapper=True), seed=0)

    assert isinstance(env, factory.RawRoboCasaAdapter)


def test_raw_adapter_keeps_fixed_obs_shape_when_payload_shape_changes():
    env = factory.RawRoboCasaAdapter(raw_env=_VariableObsRawEnv(), horizon=5)

    assert env.observation_space.shape == (192,)

    obs, _ = env.reset(seed=0)
    assert obs.shape == (192,)

    obs, _, _, _, _ = env.step(np.zeros(1, dtype=np.float32))
    assert obs.shape == (192,)
