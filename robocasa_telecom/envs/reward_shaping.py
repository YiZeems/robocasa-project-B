"""Distance-based reward shaping wrapper for RoboCasa.

RoboCasa's built-in reward function is purely sparse (0 or 1), and the
``reward_shaping`` parameter is marked as unused in the current codebase.
This wrapper adds a dense component based on the gripper-to-door distance
that is already available in the observation dict, making the task learnable
in a reasonable number of steps.

Shaped reward formula:
    r = r_sparse + w_reach * r_reach + w_open * r_open

where:
    r_sparse  = 1.0 on task success, 0 otherwise
    r_reach   = exp(-||door_obj_to_robot0_eef_pos||) in [0, 1]
                  (1 = gripper is on the door, 0 = gripper is far away)
    r_open    = hinge joint angle progress (if available), else 0

The wrapper sits between the raw RoboCasa env and RawRoboCasaAdapter so it
operates on dict observations and forwards the enriched reward downstream.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class RewardShapingWrapper:
    """Thin wrapper around a raw robosuite/robocasa env that enriches rewards.

    Designed to be composed *before* RawRoboCasaAdapter so it works on the
    raw dict observations returned by robosuite.make().
    """

    def __init__(
        self,
        env: Any,
        w_reach: float = 0.3,
        w_open: float = 0.5,
    ):
        """
        Args:
            env:      Raw robosuite environment (from robosuite.make()).
            w_reach:  Weight for the gripper-approach reward component.
            w_open:   Weight for the door-opening progress component.
        """
        self._env = env
        self._w_reach = float(w_reach)
        self._w_open = float(w_open)
        self._prev_open_angle: float = 0.0

    # ------------------------------------------------------------------
    # Gymnasium-compatible API forwarded to raw env
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None):
        self._prev_open_angle = 0.0
        try:
            out = self._env.reset(seed=seed)
        except TypeError:
            out = self._env.reset()
        return out

    def step(self, action: np.ndarray):
        out = self._env.step(action)

        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            shaped = self._shape_reward(float(reward), obs)
            return obs, shaped, terminated, truncated, info
        elif len(out) == 4:
            obs, reward, done, info = out
            shaped = self._shape_reward(float(reward), obs)
            return obs, shaped, done, info
        else:
            raise RuntimeError(f"Unexpected step() tuple length: {len(out)}")

    def close(self):
        self._env.close()

    def render(self):
        return self._env.render()

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _shape_reward(self, sparse_reward: float, obs: Any) -> float:
        """Add dense components on top of the sparse task reward."""
        r_reach = self._reach_reward(obs)
        r_open = self._open_reward(obs)
        return sparse_reward + self._w_reach * r_reach + self._w_open * r_open

    def _reach_reward(self, obs: Any) -> float:
        """Reward for the gripper approaching the door handle.

        Uses ``door_obj_to_robot0_eef_pos`` if available in the observation
        dict — it is a 3-vector pointing from the door to the end-effector.
        Falls back to 0 if the key is absent (e.g. after flattening).
        """
        if not isinstance(obs, dict):
            return 0.0
        vec = obs.get("door_obj_to_robot0_eef_pos")
        if vec is None:
            return 0.0
        dist = float(np.linalg.norm(np.asarray(vec, dtype=np.float32)))
        # Exponential kernel: reward = 1 when dist=0, decays to ~0 at dist=1m
        return float(np.exp(-5.0 * dist))

    def _open_reward(self, obs: Any) -> float:
        """Reward for increasing door opening angle.

        Tries to read the hinge joint angle from the raw env's sim. Falls
        back to 0 if the joint cannot be found — this component is a bonus,
        not required for basic learning.
        """
        try:
            sim = self._env.sim
            # RoboCasa door hinge joint names vary; try common patterns.
            for candidate in ("hinge_joint", "door_hinge", "cabinet_hinge", "fridge_hinge"):
                try:
                    joint_id = sim.model.joint_name2id(candidate)
                    angle = float(sim.data.qpos[sim.model.jnt_qposadr[joint_id]])
                    delta = max(0.0, angle - self._prev_open_angle)
                    self._prev_open_angle = angle
                    return delta
                except Exception:
                    continue
        except Exception:
            pass
        return 0.0

    # ------------------------------------------------------------------
    # Passthrough for attributes RawRoboCasaAdapter accesses
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)
