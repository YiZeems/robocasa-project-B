from __future__ import annotations

import json

from robocasa_telecom.utils.checkpoints import (
    checkpoint_sidecars,
    find_latest_resume_candidate,
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


def test_find_latest_resume_candidate_skips_completed_run(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()

    completed = checkpoint_root / "OpenCabinet_SAC_seed0_20260101_000000"
    completed.mkdir()
    completed_model = completed / "sac_100_steps.zip"
    completed_model.write_bytes(b"done")
    (completed / "final_model.zip").write_bytes(b"final")
    with (completed / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"completed_timesteps": 100, "total_timesteps": 100}, f)

    interrupted = checkpoint_root / "OpenCabinet_SAC_seed0_20260101_010000"
    interrupted.mkdir()
    resume_model = interrupted / "sac_250_steps.zip"
    resume_model.write_bytes(b"resume")
    save_checkpoint_metadata(
        resume_model,
        run_id=interrupted.name,
        algorithm="SAC",
        num_timesteps=250,
        source="periodic",
        save_replay_buffer=False,
    )

    candidate = find_latest_resume_candidate(checkpoint_root, "OpenCabinet_SAC_seed0_")

    assert candidate is not None
    assert candidate.run_dir == interrupted.resolve()
    assert candidate.artifact.model_path == resume_model.resolve()
