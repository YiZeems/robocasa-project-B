PYTHON_VERSION ?= 3.11
SEED ?= 0
CONFIG ?= configs/train/open_single_door_ppo.yaml
CHECKPOINT ?= checkpoints/<run_id>/best_model.zip
EPISODES ?= 20
PLOT_RUNS ?= outputs/OpenCabinet_SAC_seed0_*/
SMOOTH ?= 3
VIDEO_OUT ?= outputs/eval/videos
VIDEO_FPS ?= 20

.PHONY: setup sanity check tensorboard \
        train train-sac-debug train-sac train-sac-tuned train-ppo-baseline \
        eval eval-validation eval-test \
        eval-video \
        render-best-run plot \
        slurm-train slurm-eval slurm-render-best-run

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

eval-video:
	uv run python -m robocasa_telecom.rl.eval_video \
	  --config $(CONFIG) --checkpoint $(CHECKPOINT) \
	  --episodes $(EPISODES) --seed $(SEED) \
	  --out $(VIDEO_OUT) --fps $(VIDEO_FPS)

render-best-run:
	uv run python -m robocasa_telecom.render_best_run \
	  --config $(CONFIG) --checkpoint $(CHECKPOINT) --seed $(SEED)

plot:
	uv run python scripts/plot_training.py \
	  --run $(PLOT_RUNS) --smooth $(SMOOTH) --out outputs/plots

slurm-train:
	sbatch --export=ALL,CONFIG_PATH=$(CONFIG) scripts/slurm/train_array.sbatch

slurm-eval:
	sbatch --export=ALL,CONFIG_PATH=$(CONFIG),CHECKPOINT_PATH=$(CHECKPOINT) \
	  scripts/slurm/eval.sbatch

slurm-render-best-run:
	sbatch --export=ALL,CONFIG_PATH=$(CONFIG),CHECKPOINT_PATH=$(CHECKPOINT),SEED=$(SEED) \
	  scripts/slurm/render_best_run.sbatch
