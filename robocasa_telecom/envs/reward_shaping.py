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
    r_reach   = exp(-5 * dist_to_handle) in [0, 1]
    r_open    = normalized door joint angle in [0, 1]

Per-step debug values are written into the ``info`` dict under keys prefixed
with ``_dbg_`` so they can be consumed by debug_reward.py without affecting
the training signal.

Handle body resolution
----------------------
``fxtr.handle_name`` is unreliable — it returns a ``_handle_handle`` suffix
that does not exist in MuJoCo, or ``?`` when the fixture has no name.
We search ``sim.model.body_names`` once per episode at reset time instead.

Joint resolution
----------------
``get_joint_state(env, joint_names)`` always requires an explicit
``joint_names`` list.  We obtain it from ``fxtr.door_joint_names`` (which
filters ``fxtr._joint_infos`` by the substring ``"door"``) and cache it per
episode.  Fallbacks cover edge cases where the property is unavailable.
For double-door cabinets the joint matching the selected handle side
(left / right) is used; for single-door cabinets the one joint is used
directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# MuJoCo helpers (compatible with both mujoco-py and newer mujoco APIs)
# ---------------------------------------------------------------------------

def _all_body_names(sim: Any) -> list[str]:
    """Return non-empty body names from the sim model."""
    try:
        return [n for n in sim.model.body_names if n]
    except AttributeError:
        pass
    try:
        return [sim.model.body(i).name for i in range(sim.model.nbody) if sim.model.body(i).name]
    except Exception:
        return []


def _body_pos(sim: Any, name: str) -> np.ndarray:
    """Return body world position as float32 (3,) array."""
    try:
        bid = sim.model.body_name2id(name)
    except Exception:
        bid = sim.model.body(name).id
    return np.array(sim.data.body_xpos[bid], dtype=np.float32)


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

class RewardShapingWrapper:
    """Thin wrapper around a raw robosuite/robocasa env that enriches rewards.

    Designed to be composed *before* RawRoboCasaAdapter so it works on the
    raw dict observations returned by robosuite.make().
    """

    def __init__(
        self,
        env: Any,
        w_reach: float = 0.1,
        w_contact: float = 0.0,
        w_open: float = 0.3,
        w_delta: float = 10.0,
        w_grasp: float = 5.0,
        contact_threshold: float = 0.10,
        gripper_open_max: float = 0.08,
    ):
        self._env = env
        self._w_reach = float(w_reach)
        self._w_contact = float(w_contact)
        self._w_open = float(w_open)
        self._w_delta = float(w_delta)
        self._w_grasp = float(w_grasp)
        self._contact_threshold = float(contact_threshold)
        # Panda gripper: each finger in [0, ~0.04]m; fully open sum ≈ 0.08
        self._gripper_open_max = float(gripper_open_max)
        # Both resolved once per episode at reset(); empty / None = disabled.
        self._handle_body: str | None = None
        self._joint_names: list[str] = []
        # Cached per-step state consumed by ObsAugmentWrapper.
        self._last_handle_pos: np.ndarray | None = None
        self._last_eef_pos: np.ndarray | None = None
        self._last_open: float = 0.0
        self._last_dist: float = 0.0
        self._prev_open: float = 0.0   # for delta-open computation
        # Log only on the first reset per worker — avoids flooding the terminal
        # when 8 SubprocVecEnv workers each reset thousands of times.
        self._reset_count: int = 0

    # ------------------------------------------------------------------
    # Gymnasium-compatible API forwarded to raw env
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None):
        self._handle_body = None
        self._joint_names = []
        self._last_handle_pos = None
        self._last_eef_pos = None
        self._last_open = 0.0
        self._last_dist = 0.0
        self._prev_open = 0.0
        try:
            out = self._env.reset(seed=seed)
        except TypeError:
            out = self._env.reset()
        first_obs = out[0] if isinstance(out, tuple) else out
        self._handle_body = self._resolve_handle_body(first_obs)
        self._joint_names = self._resolve_joint_names()
        self._reset_count += 1
        if self._reset_count == 1:
            # Log once per worker — enough to verify wiring without flooding
            # the terminal across 8+ SubprocVecEnv workers over thousands of resets.
            self._log_episode_start()
        return out

    def step(self, action: np.ndarray):
        out = self._env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            info = dict(info or {})
            shaped = self._shape_reward(float(reward), obs, info)
            return obs, shaped, terminated, truncated, info
        if len(out) == 4:
            obs, reward, done, info = out
            info = dict(info or {})
            shaped = self._shape_reward(float(reward), obs, info)
            return obs, shaped, done, info
        raise RuntimeError(f"Unexpected step() tuple length: {len(out)}")

    def close(self):
        self._env.close()

    def render(self):
        return self._env.render()

    # ------------------------------------------------------------------
    # Resolution helpers (called once at reset)
    # ------------------------------------------------------------------

    def _resolve_handle_body(self, obs: Any) -> str | None:
        """Search sim body names for the door handle; return name or None."""
        try:
            sim = self._env.sim
        except Exception:
            return None

        fxtr = getattr(self._env, "fxtr", None)
        fxtr_name = ""
        if fxtr is not None:
            raw = getattr(fxtr, "name", "") or ""
            if raw and raw != "?":
                fxtr_name = raw

        all_bodies = _all_body_names(sim)
        if not all_bodies:
            return None

        candidates = [b for b in all_bodies if "handle" in b.lower()]

        if fxtr_name:
            prefix = [b for b in candidates if b.startswith(fxtr_name)]
            if prefix:
                candidates = prefix

        if not candidates:
            return None

        main = [b for b in candidates if b.endswith("_handle_main")]
        if main:
            candidates = main

        if len(candidates) == 1:
            return candidates[0]

        # Multiple candidates: pick closest to EEF at reset.
        eef_pos: np.ndarray | None = None
        if isinstance(obs, dict) and "robot0_eef_pos" in obs:
            eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
        if eef_pos is not None:
            best: str | None = None
            best_d = float("inf")
            for cand in candidates:
                try:
                    d = float(np.linalg.norm(_body_pos(sim, cand) - eef_pos))
                    if d < best_d:
                        best_d = d
                        best = cand
                except Exception:
                    pass
            if best is not None:
                return best

        return sorted(candidates)[0]

    def _resolve_joint_names(self) -> list[str]:
        """Return door joint names for this fixture.

        Priority:
        1. ``fxtr.door_joint_names`` property (base Fixture class filters
           ``_joint_infos`` keys by ``"door"`` substring — covers both
           ``{name}_doorhinge`` and ``{name}_leftdoorhinge`` / ``{name}_rightdoorhinge``).
        2. Scan ``fxtr._joint_infos`` for keys containing ``"hinge"`` or ``"slide"``.
        3. Scan ``sim.model.joint_names`` for joints starting with the fixture
           prefix and containing a hinge/door keyword.
        """
        fxtr = getattr(self._env, "fxtr", None)
        if fxtr is None:
            return []

        # --- Primary: door_joint_names property ---
        try:
            jnames = list(fxtr.door_joint_names)
            if jnames:
                return jnames
        except Exception:
            pass

        # --- Fallback 1: scan _joint_infos ---
        try:
            keywords = ("hinge", "door", "slide")
            jnames = [j for j in fxtr._joint_infos if any(kw in j.lower() for kw in keywords)]
            if jnames:
                return jnames
        except Exception:
            pass

        # --- Fallback 2: scan sim model joint names by fixture prefix ---
        try:
            fxtr_name = (getattr(fxtr, "name", "") or "").strip()
            sim = self._env.sim
            keywords = ("hinge", "door", "slide")
            all_joints = list(sim.model.joint_names)
            jnames = [
                j for j in all_joints
                if (not fxtr_name or j.startswith(fxtr_name))
                and any(kw in j.lower() for kw in keywords)
            ]
            if jnames:
                return jnames
        except Exception:
            pass

        return []

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _shape_reward(self, sparse_reward: float, obs: Any, info: dict) -> float:
        r_reach, dist = self._reach_reward(obs)
        r_open, open_norm = self._open_reward()

        # Contact bonus: disabled (w_contact=0.0) — kept for future experiments.
        r_contact = (
            1.0 if (not np.isnan(dist) and dist < self._contact_threshold) else 0.0
        )

        # Delta-open: reward each increment of door-opening progress this step.
        r_delta = max(0.0, float(open_norm) - float(self._prev_open))
        self._prev_open = float(open_norm)

        # Grasp-delta: reward closing the gripper ONLY when the door is also moving.
        # Pure r_grasp without this condition caused "grasp-and-flee" abuse —
        # the agent learned to close its gripper near the handle then immediately
        # retreat, collecting the grasp bonus without opening anything.
        # Multiplying by r_delta ensures the signal only fires when closing the
        # gripper actually produces door movement.
        r_grasp = self._gripper_reward(obs, dist)
        r_grasp_delta = r_grasp * r_delta  # non-zero only if closing AND door moving

        shaped = (
            sparse_reward
            + self._w_reach * r_reach
            + self._w_contact * r_contact
            + self._w_open * r_open
            + self._w_delta * r_delta
            + self._w_grasp * r_grasp_delta
        )

        self._last_open = open_norm
        self._last_dist = dist

        info["_dbg_r_sparse"] = sparse_reward
        info["_dbg_r_reach"] = r_reach
        info["_dbg_r_contact"] = r_contact
        info["_dbg_r_open"] = r_open
        info["_dbg_r_delta"] = r_delta
        info["_dbg_r_grasp"] = r_grasp          # raw closure ∈ [0,1]
        info["_dbg_r_grasp_delta"] = r_grasp_delta  # closure × Δopen (what's actually rewarded)
        info["_dbg_open"] = open_norm
        info["_dbg_dist"] = dist
        return shaped

    def _reach_reward(self, obs: Any) -> tuple[float, float]:
        """Return (r_reach, dist) using cached handle body.

        Returns (0.0, NaN) when handle is unresolved so callers can distinguish
        "genuinely zero distance" from "no handle found this episode".
        """
        if self._handle_body is None:
            return 0.0, float("nan")
        try:
            sim = self._env.sim
            handle_pos = _body_pos(sim, self._handle_body)
            if isinstance(obs, dict) and "robot0_eef_pos" in obs:
                eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
                self._last_handle_pos = handle_pos
                self._last_eef_pos = eef_pos
                dist = float(np.linalg.norm(handle_pos - eef_pos))
                return float(np.exp(-5.0 * dist)), dist
        except Exception:
            pass
        return 0.0, 0.0

    def _gripper_reward(self, obs: Any, dist: float) -> float:
        """Reward gripper closure when the EEF is within contact range of the handle.

        Panda gripper: robot0_gripper_qpos = [left_finger, right_finger] in metres.
        Fully open: each finger ≈ 0.04 m → sum ≈ 0.08.
        Fully closed: both ≈ 0.0 → sum ≈ 0.0.
        closure ∈ [0, 1]: 1 = fully closed, 0 = fully open.

        Only active when dist < contact_threshold so the agent is not rewarded
        for closing the gripper far from the handle.
        """
        if np.isnan(dist) or dist > self._contact_threshold:
            return 0.0
        if not isinstance(obs, dict) or "robot0_gripper_qpos" not in obs:
            return 0.0
        qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
        qpos_sum = float(np.sum(np.abs(qpos)))
        closure = max(0.0, 1.0 - qpos_sum / self._gripper_open_max)
        return closure

    def _open_reward(self) -> tuple[float, float]:
        """Return (r_open, open_norm) using cached joint names.

        For single-door cabinets there is one joint; for double-door cabinets
        we pick the joint whose side (left/right) matches the selected handle.
        Falls back to max over all joints when side matching is ambiguous.
        """
        if not self._joint_names:
            return 0.0, 0.0
        try:
            fxtr = getattr(self._env, "fxtr", None)
            if fxtr is None:
                return 0.0, 0.0
            joint_state = fxtr.get_joint_state(self._env, self._joint_names)
            if not joint_state:
                return 0.0, 0.0
            open_norm = self._select_open_value(joint_state)
            return open_norm, open_norm
        except Exception:
            pass
        return 0.0, 0.0

    def _select_open_value(self, joint_state: dict[str, float]) -> float:
        """Pick the relevant open value from a joint_state dict.

        Single joint → return it directly.
        Multiple joints → try to match the side (left/right) of the selected
        handle body; fall back to max over all values.
        """
        if len(joint_state) == 1:
            return float(next(iter(joint_state.values())))

        if self._handle_body:
            handle_lower = self._handle_body.lower()
            for side in ("right", "left"):
                if side in handle_lower:
                    side_matches = {k: v for k, v in joint_state.items() if side in k.lower()}
                    if side_matches:
                        return float(next(iter(side_matches.values())))

        return float(max(joint_state.values()))

    # ------------------------------------------------------------------
    # Episode-start logging
    # ------------------------------------------------------------------

    def _log_episode_start(self) -> None:
        """Emit a full fixture/handle/joint diagnostic line at each episode start."""
        try:
            sim = self._env.sim
            fxtr = getattr(self._env, "fxtr", None)
            fxtr_name = (getattr(fxtr, "name", None) or "?") if fxtr else "?"

            # Handle body diagnostics
            all_bodies = _all_body_names(sim)
            all_handle_cands = [b for b in all_bodies if "handle" in b.lower()]
            prefix_cands = (
                [b for b in all_handle_cands if b.startswith(fxtr_name)]
                if fxtr_name and fxtr_name != "?"
                else []
            )

            print(f"[RewardShaping] fixture_name     = {fxtr_name!r}")
            print(f"[RewardShaping] handle candidates = {all_handle_cands}")
            if prefix_cands:
                print(f"[RewardShaping] prefix-filtered   = {prefix_cands}")
            print(f"[RewardShaping] selected handle   = {self._handle_body!r}")

            if self._handle_body:
                try:
                    pos = _body_pos(sim, self._handle_body)
                    print(f"[RewardShaping] handle_pos        = {tuple(round(float(x), 3) for x in pos)}")
                except Exception as e:
                    print(f"[RewardShaping] handle_pos ERROR  = {e}")
            else:
                print("[RewardShaping] WARNING: no handle body resolved — r_reach = 0 this episode")

            # Joint / open diagnostics
            print(f"[RewardShaping] door_joint_names  = {self._joint_names}")
            if self._joint_names and fxtr is not None:
                try:
                    js = fxtr.get_joint_state(self._env, self._joint_names)
                    open_vals = {k: round(v, 3) for k, v in js.items()} if js else {}
                    print(f"[RewardShaping] open amounts      = {open_vals}")
                except Exception as e:
                    print(f"[RewardShaping] joint_state ERROR = {e}")
            elif not self._joint_names:
                print("[RewardShaping] WARNING: no door joints resolved — r_open = 0 this episode")

        except Exception as e:
            print(f"[RewardShaping] _log_episode_start ERROR: {e}")

    # ------------------------------------------------------------------
    # Passthrough
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)
