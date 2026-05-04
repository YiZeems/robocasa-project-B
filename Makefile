PYTHON_VERSION ?= 3.11
SEED ?= 0
CONFIG ?= configs/train/open_single_door_ppo.yaml
CHECKPOINT ?= checkpoints/<run_id>/best_model.zip
EPISODES ?= 20

.PHONY: setup sanity check tensorboard \
        train train-sac-debug train-sac train-sac-tuned train-ppo-baseline \
        eval eval-validation eval-test \
        slurm-train slurm-eval

setup:
	PYTHON_VERSION=$(PYTHON_VERSION) bash scripts/setup_uv.sh

sanity:
	uv run python -m robocasa_telecom.sanity --config configs/env/open_single_door.yaml --steps 20

check:
	uv pip check
	uv run python -m compileall robocasa_telecom tests

tensorboard:
	uv run tensorboard --logdir logs/tensorboard

train:
	uv run python -m robocasa_telecom.train --config $(CONFIG) --seed $(SEED)

train-sac-debug:
	uv run python -m robocasa_telecom.train \
	  --config configs/train/open_single_door_sac_debug.yaml --seed $(SEED)

train-sac:
	uv run python -m robocasa_telecom.train \
	  --config configs/train/open_single_door_sac.yaml --seed $(SEED)

train-sac-tuned:
	uv run python -m robocasa_telecom.train \
	  --config configs/train/open_single_door_sac_tuned.yaml --seed $(SEED)

train-ppo-baseline:
	uv run python -m robocasa_telecom.train \
	  --config configs/train/open_single_door_ppo_baseline.yaml --seed $(SEED)

eval:
	uv run python -m robocasa_telecom.evaluate \
	  --config $(CONFIG) --checkpoint $(CHECKPOINT) \
	  --num-episodes $(EPISODES) --deterministic

eval-validation:
	uv run python -m robocasa_telecom.evaluate \
	  --config $(CONFIG) --checkpoint $(CHECKPOINT) \
	  --num-episodes $(EPISODES) --split validation --deterministic

eval-test:
	uv run python -m robocasa_telecom.evaluate \
	  --config $(CONFIG) --checkpoint $(CHECKPOINT) \
	  --num-episodes $(EPISODES) --split test --deterministic

slurm-train:
	sbatch --export=ALL,CONFIG_PATH=$(CONFIG) scripts/slurm/train_array.sbatch

slurm-eval:
	sbatch --export=ALL,CONFIG_PATH=$(CONFIG),CHECKPOINT_PATH=$(CHECKPOINT) \
	  scripts/slurm/eval.sbatch
