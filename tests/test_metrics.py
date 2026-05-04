from __future__ import annotations

import math

from robocasa_telecom.utils.metrics import (
    action_magnitude,
    extract_scalar_metric,
    prefixed_metrics,
    summarize_rollout_episodes,
)


class _DummyModel:
    def predict(self, obs, deterministic=True):
        del obs, deterministic
        return [0.0, 0.0], None


class _DummyEnv:
    def __init__(self):
        self._episode = 0
        self._step = 0
        self.door_angle = 0.0

    def reset(self, seed=None):
        del seed
        self._episode += 1
        self._step = 0
        self.door_angle = 0.0
        return [0.0], {}

    def step(self, action):
        del action
        self._step += 1
        self.door_angle = float(self._step * 10)
        terminated = self._step >= 2
        info = {"metrics": {"door_angle": self.door_angle}}
        if terminated:
            info["success"] = True
        return [0.0], 1.0, terminated, False, info


def test_prefixed_metrics_filters_non_numeric_values():
    metrics = prefixed_metrics(
        {
            "return_mean": 1.5,
            "task": "OpenCabinet",
            "episode_length_mean": 42,
        },
        "train_",
    )

    assert metrics == {
        "train_return_mean": 1.5,
        "train_episode_length_mean": 42.0,
    }


def test_extract_scalar_metric_reads_nested_mapping():
    info = {"metrics": {"door_angle": 12.5}}

    assert extract_scalar_metric(info, None, ("door_angle",)) == 12.5


def test_summarize_rollout_episodes_collects_progress_metrics():
    metrics = summarize_rollout_episodes(_DummyModel(), _DummyEnv(), episodes=2)

    assert metrics["return_mean"] == 2.0
    assert metrics["success_rate"] == 1.0
    assert metrics["episode_length_mean"] == 2.0
    assert metrics["action_magnitude_mean"] == 0.0
    assert math.isclose(metrics["door_angle_final_mean"], 20.0)


def test_action_magnitude_handles_vectors():
    assert action_magnitude([3.0, 4.0]) == 5.0
