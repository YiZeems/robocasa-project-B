"""Anti-hacking reward shaping for RoboCasa manipulation tasks.

Design principles:
- Progress uses a high-watermark (best door angle ever in the episode) so the
  agent cannot gain reward by oscillating the door back and forth.
- The approach component is small and disabled once the door is open, preventing
  "hover hacking" where the robot stays near the handle without acting.
- Stagnation is penalised only when the robot is already close to the handle but
  has made no new progress for N consecutive steps (conditional penalty).
- Oscillation is penalised when the door angle alternates direction repeatedly
  inside a sliding window without net progress.
- The success bonus dominates all other components so the agent always prefers
  completing the task over gaming intermediate rewards.

Reward formula:
  R = w_approach * r_approach  (gated: disabled once door open)
    + w_progress * r_progress  (high-watermark delta, anti-oscillation)
    + w_success  * r_success   (sparse, dominates)
    - w_action_reg * ||a||²    (action regularisation)
    - w_stagnation * p_stag    (conditional: close + stuck)
    - w_wrong_dir  * p_back    (door moved backward from HWM)
    - w_oscillation * p_osc    (sign changes in sliding window)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RewardConfig:
    """Hyperparameters for AntiHackingReward, loadable from YAML."""

    # Component weights
    w_approach: float = 0.05
    w_progress: float = 1.0
    w_success: float = 5.0
    w_action_reg: float = 0.01
    w_stagnation: float = 0.05
    w_wrong_dir: float = 0.3
    w_oscillation: float = 0.2

    # Thresholds
    theta_success: float = 0.90   # normalised [0,1] — matches fxtr.is_open(th=0.90)
    d_max: float = 0.5            # approach reward → 0 beyond this (m)
    d_prox: float = 0.12          # "close to handle" threshold for stagnation (m)
    stagnation_n: int = 50        # steps without progress before stagnation fires
    tol_wrong_dir: float = 0.02   # tolerance below HWM before wrong-dir penalty
    oscillation_window: int = 20  # sliding window size for oscillation detection
    oscillation_threshold: int = 4  # min sign changes in window to fire penalty

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "RewardConfig":
        """Build from a plain dict (e.g., loaded from YAML)."""
        valid = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in cfg.items() if k in valid})


class AntiHackingReward:
    """Stateful per-episode reward shaper.

    One instance per environment worker; call reset() at the start of each
    episode and compute() at every step.

    All theta values are normalised [0, 1]:
      0 = door fully closed, 1 = door fully open.
    """

    def __init__(self, cfg: RewardConfig | None = None):
        self.cfg = cfg or RewardConfig()
        self.reset()

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.theta_best: float = 0.0
        self.stagnation_ctr: int = 0
        self._prev_theta: float = 0.0
        self._delta_history: deque[float] = deque(maxlen=self.cfg.oscillation_window)
        # Diagnostic counters (exposed in episode_summary)
        self._oscillation_steps: int = 0
        self._stagnation_steps: int = 0
        self._wrong_dir_steps: int = 0
        # Accumulators for episode-level MLflow logging
        self._ep_components: dict[str, float] = {
            "approach": 0.0,
            "progress": 0.0,
            "success": 0.0,
            "action_reg": 0.0,
            "stagnation": 0.0,
            "wrong_dir": 0.0,
            "oscillation": 0.0,
            "total": 0.0,
        }

    def episode_summary(self) -> dict[str, float]:
        """Return accumulated component totals + diagnostic counters."""
        summary = dict(self._ep_components)
        summary["oscillation_steps"] = float(self._oscillation_steps)
        summary["stagnation_steps"]  = float(self._stagnation_steps)
        summary["wrong_dir_steps"]   = float(self._wrong_dir_steps)
        summary["theta_best"]        = float(self.theta_best)
        return summary

    # ------------------------------------------------------------------
    # Per-step computation
    # ------------------------------------------------------------------

    def compute(
        self,
        theta: float,
        d_ee_handle: float,
        action: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        """Compute shaped reward and component breakdown.

        Args:
            theta:       Normalised door opening in [0, 1].
            d_ee_handle: Distance EEF → handle (m).
            action:      Raw action vector from the policy.

        Returns:
            (total_reward, component_dict)
        """
        cfg = self.cfg

        # ---- High-watermark progress ----------------------------------
        delta_progress = max(0.0, theta - self.theta_best)
        if delta_progress > 1e-5:
            self.theta_best = theta
            self.stagnation_ctr = 0
        else:
            self.stagnation_ctr += 1

        # ---- Oscillation detection (sliding window of Δθ signs) ------
        raw_delta = theta - self._prev_theta
        self._prev_theta = theta
        self._delta_history.append(raw_delta)
        sign_changes = 0
        hist = list(self._delta_history)
        for i in range(1, len(hist)):
            if abs(hist[i]) > 1e-4 and abs(hist[i - 1]) > 1e-4:
                if hist[i] * hist[i - 1] < 0:
                    sign_changes += 1
        oscillating = (
            sign_changes >= cfg.oscillation_threshold
            and delta_progress < 1e-5
        )

        # ---- Component 1: approach (gated, saturated) ----------------
        if theta < cfg.theta_success:
            r_approach = float(np.clip(1.0 - d_ee_handle / cfg.d_max, 0.0, 1.0))
        else:
            r_approach = 0.0

        # ---- Component 2: progress (high-watermark, normalised) ------
        r_progress = delta_progress / max(cfg.theta_success, 1e-8)

        # ---- Component 3: success bonus (sparse) ---------------------
        r_success = 1.0 if theta >= cfg.theta_success else 0.0

        # ---- Component 4: action regularisation ----------------------
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        r_action_reg = -float(np.dot(action_arr, action_arr))

        # ---- Component 5: stagnation penalty (conditional) ----------
        if d_ee_handle < cfg.d_prox and self.stagnation_ctr >= cfg.stagnation_n:
            r_stagnation = -1.0
            self._stagnation_steps += 1
        else:
            r_stagnation = 0.0

        # ---- Component 6: wrong-direction penalty -------------------
        if theta < self.theta_best - cfg.tol_wrong_dir:
            r_wrong_dir = -(self.theta_best - theta)
            self._wrong_dir_steps += 1
        else:
            r_wrong_dir = 0.0

        # ---- Component 7: oscillation penalty -----------------------
        if oscillating:
            r_oscillation = -float(sign_changes) / max(cfg.oscillation_window, 1)
            self._oscillation_steps += 1
        else:
            r_oscillation = 0.0

        # ---- Weighted sum -------------------------------------------
        total = (
            cfg.w_approach    * r_approach
            + cfg.w_progress    * r_progress
            + cfg.w_success     * r_success
            + cfg.w_action_reg  * r_action_reg
            + cfg.w_stagnation  * r_stagnation
            + cfg.w_wrong_dir   * r_wrong_dir
            + cfg.w_oscillation * r_oscillation
        )

        components: dict[str, float] = {
            # Weighted contributions
            "approach":    cfg.w_approach    * r_approach,
            "progress":    cfg.w_progress    * r_progress,
            "success":     cfg.w_success     * r_success,
            "action_reg":  cfg.w_action_reg  * r_action_reg,
            "stagnation":  cfg.w_stagnation  * r_stagnation,
            "wrong_dir":   cfg.w_wrong_dir   * r_wrong_dir,
            "oscillation": cfg.w_oscillation * r_oscillation,
            "total":       total,
            # Raw diagnostic signals
            "theta":          float(theta),
            "theta_best":     float(self.theta_best),
            "d_ee_handle":    float(d_ee_handle),
            "stagnation_ctr": float(self.stagnation_ctr),
            "sign_changes":   float(sign_changes),
        }

        for key in ("approach", "progress", "success", "action_reg",
                    "stagnation", "wrong_dir", "oscillation", "total"):
            self._ep_components[key] += components[key]

        return total, components
