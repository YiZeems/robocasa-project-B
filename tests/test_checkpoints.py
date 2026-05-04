from __future__ import annotations

import json

from robocasa_telecom.utils.checkpoints import (
    checkpoint_sidecars,
    resolve_checkpoint_artifact,
    resolve_checkpoint_path,
    save_checkpoint_metadata,
)


def test_resolve_checkpoint_path_picks_latest_step(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    older = run_dir / "sac_100_steps.zip"
    newer = run_dir / "sac_250_steps.zip"
    older.write_bytes(b"older")
    newer.write_bytes(b"newer")

    assert resolve_checkpoint_path(run_dir) == newer.resolve()


def test_checkpoint_sidecars_follow_model_path(tmp_path):
    model_path = tmp_path / "sac_250_steps.zip"
    replay_path, metadata_path = checkpoint_sidecars(model_path)

    assert replay_path.name == "sac_250_steps_replay_buffer.pkl"
    assert metadata_path.name == "sac_250_steps.json"


def test_resolve_checkpoint_artifact_from_metadata_json(tmp_path):
    model_path = tmp_path / "sac_500_steps.zip"
    model_path.write_bytes(b"model")
    metadata_path = save_checkpoint_metadata(
        model_path,
        run_id="run",
        algorithm="SAC",
        num_timesteps=500,
        source="periodic",
        save_replay_buffer=True,
    )

    with metadata_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["checkpoint_path"] == str(model_path.resolve())
    assert resolve_checkpoint_artifact(metadata_path).model_path == model_path.resolve()
