from __future__ import annotations

import pytest

from robocasa_telecom.rl.train import _env_flag, _env_int


def test_env_flag_parses_truthy_and_falsy(monkeypatch):
    monkeypatch.setenv("ROBOCASA_RENDER_BEST_RUN_VIDEO", "1")
    assert _env_flag("ROBOCASA_RENDER_BEST_RUN_VIDEO") is True

    monkeypatch.setenv("ROBOCASA_RENDER_BEST_RUN_VIDEO", "0")
    assert _env_flag("ROBOCASA_RENDER_BEST_RUN_VIDEO") is False


def test_env_flag_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv("ROBOCASA_RENDER_BEST_RUN_VIDEO", "maybe")
    with pytest.raises(ValueError):
        _env_flag("ROBOCASA_RENDER_BEST_RUN_VIDEO")


def test_env_int_parses_and_defaults(monkeypatch):
    monkeypatch.delenv("ROBOCASA_RENDER_BEST_RUN_VIDEO_MAX_STEPS", raising=False)
    assert _env_int("ROBOCASA_RENDER_BEST_RUN_VIDEO_MAX_STEPS", 500) == 500

    monkeypatch.setenv("ROBOCASA_RENDER_BEST_RUN_VIDEO_MAX_STEPS", "120")
    assert _env_int("ROBOCASA_RENDER_BEST_RUN_VIDEO_MAX_STEPS", 500) == 120


def test_env_int_supports_video_duration_settings(monkeypatch):
    monkeypatch.setenv("ROBOCASA_RENDER_BEST_RUN_VIDEO_MIN_SECONDS", "12")
    assert _env_int("ROBOCASA_RENDER_BEST_RUN_VIDEO_MIN_SECONDS", 10) == 12

    monkeypatch.setenv("ROBOCASA_RENDER_BEST_RUN_VIDEO_MAX_EPISODES", "4")
    assert _env_int("ROBOCASA_RENDER_BEST_RUN_VIDEO_MAX_EPISODES", 5) == 4
