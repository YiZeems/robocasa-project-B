PYTHON_VERSION ?= 3.11

.PHONY: setup train eval sanity check

setup:
	PYTHON_VERSION=$(PYTHON_VERSION) bash scripts/setup_uv.sh

train:
	uv run python -m robocasa_telecom.train --config configs/train/open_single_door_ppo.yaml --seed 0

eval:
	uv run python -m robocasa_telecom.evaluate --config configs/train/open_single_door_ppo.yaml --checkpoint checkpoints/<run_id>/final_model.zip --num-episodes 20 --deterministic

sanity:
	uv run python -m robocasa_telecom.sanity --config configs/env/open_single_door.yaml --steps 20

check:
	uv run python -m pip check
	uv run python -m compileall robocasa_telecom tests
