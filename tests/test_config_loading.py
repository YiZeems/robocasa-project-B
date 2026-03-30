"""Minimal config-loading smoke test for CI and local sanity."""

from pathlib import Path

from robocasa_telecom.envs import load_env_config


def main() -> None:
    """Validate that default env config parses into expected canonical values."""

    cfg_path = Path("configs/env/open_single_door.yaml")
    env_cfg = load_env_config(cfg_path)

    assert env_cfg.robots == "PandaOmron"
    assert env_cfg.task in {"OpenCabinet", "OpenDoor", "OpenSingleDoor"}
    print("config_loading_ok")


if __name__ == "__main__":
    main()
