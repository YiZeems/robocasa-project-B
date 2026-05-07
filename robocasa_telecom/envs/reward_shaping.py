
from __future__ import annotations

from typing import Any

import numpy as np


def _all_body_names(sim: Any) -> list[str]:
    try:
        return [n for n in sim.model.body_names if n]
    except AttributeError:
        pass
    try:
        return [sim.model.body(i).name for i in range(sim.model.nbody) if sim.model.body(i).name]
    except Exception:
        return []


def _body_pos(sim: Any, name: str) -> np.ndarray:
    try:
        bid = sim.model.body_name2id(name)
    except Exception:
        bid = sim.model.body(name).id
    return np.array(sim.data.body_xpos[bid], dtype=np.float32)


class RewardShapingWrapper:

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
                                                                          
        self._gripper_open_max = float(gripper_open_max)
                                                                             
        self._handle_body: str | None = None
        self._joint_names: list[str] = []
                                                              
        self._last_handle_pos: np.ndarray | None = None
        self._last_eef_pos: np.ndarray | None = None
        self._last_open: float = 0.0
        self._last_dist: float = 0.0
        self._prev_open: float = 0.0                               
                                                                               
                                                                     
        self._reset_count: int = 0

                                                                        
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

                                                                        
    def _resolve_handle_body(self, obs: Any) -> str | None:
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
        fxtr = getattr(self._env, "fxtr", None)
        if fxtr is None:
            return []

                                                    
        try:
            jnames = list(fxtr.door_joint_names)
            if jnames:
                return jnames
        except Exception:
            pass

                                               
        try:
            keywords = ("hinge", "door", "slide")
            jnames = [j for j in fxtr._joint_infos if any(kw in j.lower() for kw in keywords)]
            if jnames:
                return jnames
        except Exception:
            pass

                                                                          
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

                                                                        
    def _shape_reward(self, sparse_reward: float, obs: Any, info: dict) -> float:
        r_reach, dist = self._reach_reward(obs)
        r_open, open_norm = self._open_reward()

                                                                                
        r_contact = (
            1.0 if (not np.isnan(dist) and dist < self._contact_threshold) else 0.0
        )

                                                                               
        r_delta = max(0.0, float(open_norm) - float(self._prev_open))
        self._prev_open = float(open_norm)

                                                                                    
        r_grasp = self._gripper_reward(obs, dist)
        r_grasp_delta = r_grasp * r_delta                                            

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
        info["_dbg_r_grasp"] = r_grasp                               
        info["_dbg_r_grasp_delta"] = r_grasp_delta                                              
        info["_dbg_open"] = open_norm
        info["_dbg_dist"] = dist
        return shaped

    def _reach_reward(self, obs: Any) -> tuple[float, float]:
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
        if np.isnan(dist) or dist > self._contact_threshold:
            return 0.0
        if not isinstance(obs, dict) or "robot0_gripper_qpos" not in obs:
            return 0.0
        qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
        qpos_sum = float(np.sum(np.abs(qpos)))
        closure = max(0.0, 1.0 - qpos_sum / self._gripper_open_max)
        return closure

    def _open_reward(self) -> tuple[float, float]:
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

                                                                        
    def _log_episode_start(self) -> None:
        try:
            sim = self._env.sim
            fxtr = getattr(self._env, "fxtr", None)
            fxtr_name = (getattr(fxtr, "name", None) or "?") if fxtr else "?"

                                     
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

                                                                        
    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)
