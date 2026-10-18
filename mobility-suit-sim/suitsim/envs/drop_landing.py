"""T1 — drop landing from 1–5 m at random body orientation (spec 5.3 / 5.4)."""
from __future__ import annotations

import math

import mujoco
import numpy as np

from suitsim.envs.base import SuitEnvBase
from suitsim.interfaces import FEET
from suitsim.model import set_initial_pose


def _axis_angle_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / max(np.linalg.norm(a), 1e-12)
    return np.concatenate([[math.cos(angle / 2)], math.sin(angle / 2) * a])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


class DropLandingEnv(SuitEnvBase):
    """Released from rest at feet height ``drop_height`` with initial tilt / angular velocity.

    ``cfg['task']`` ranges (``drop_height``, ``initial_tilt_deg``, ``initial_yaw_deg``,
    ``initial_angvel``, ``initial_linvel``) and ``cfg['randomization']`` define the episode
    distribution; ``task.fixed`` (e.g. ``{drop_height: 3, tilt_deg: 0, angvel: 0}``) pins any
    of ``drop_height, tilt_deg, tilt_axis_deg, yaw_deg, angvel (scalar|3-vector), mass_scale,
    friction, noise_scale, fmax_scale, tau_scale``.
    """

    def __init__(self, cfg: dict, controller: str = "baseline", render_mode: str | None = None,
                 log_trajectory: bool = False, policy_dt: float | None = None, **kw):
        super().__init__(cfg, controller=controller, render_mode=render_mode,
                         log_trajectory=log_trajectory, policy_dt=policy_dt, **kw)
        task = cfg["task"]
        self.settle_time = float(task.get("settle_time", 0.5))
        sl = cfg["safe_landing"]
        self.lim_v = float(sl["max_vertical_speed"])
        self.lim_h = float(sl["max_horizontal_speed"])
        self.lim_tilt = float(sl["max_tilt_deg"])
        self.rw = dict(cfg.get("reward", {}) or {})

    # ------------------------------------------------------------------
    def _sample_episode(self, rng: np.random.Generator) -> dict:
        task = self.cfg["task"]
        fixed = task.get("fixed") or {}
        # draw everything unconditionally so the random stream does not depend on `fixed`
        dh = task.get("drop_height", [1.0, 5.0])
        h = rng.uniform(*dh) if isinstance(dh, (list, tuple)) else float(dh)
        tilt = rng.uniform(0.0, float(task.get("initial_tilt_deg", 0.0)))
        axis_deg = rng.uniform(0.0, 360.0)
        ymax = float(task.get("initial_yaw_deg", 0.0))
        yaw = rng.uniform(-ymax, ymax)
        wmax = float(task.get("initial_angvel", 0.0))
        w = rng.uniform(-wmax, wmax, 3)
        vmax = float(task.get("initial_linvel", 0.0))
        v = rng.uniform(-vmax, vmax, 3)
        dr = self._apply_dr_common(rng)

        h = float(fixed.get("drop_height", h))
        tilt = float(fixed.get("tilt_deg", tilt))
        axis_deg = float(fixed.get("tilt_axis_deg", axis_deg))
        yaw = float(fixed.get("yaw_deg", yaw))
        if "angvel" in fixed:
            w = np.broadcast_to(np.asarray(fixed["angvel"], float), (3,)).copy()
        if "linvel" in fixed:
            v = np.broadcast_to(np.asarray(fixed["linvel"], float), (3,)).copy()

        ax = math.radians(axis_deg)
        q_tilt = _axis_angle_quat(np.array([math.cos(ax), math.sin(ax), 0.0]), math.radians(tilt))
        q_yaw = _axis_angle_quat(np.array([0.0, 0.0, 1.0]), math.radians(yaw))
        q = _quat_mul(q_tilt, q_yaw)
        set_initial_pose(self.model, self.data, h, q, w, linvel=v, cfg=self.cfg)
        return {"drop_height": h, "tilt_deg": tilt, "tilt_axis_deg": axis_deg, "yaw_deg": yaw,
                "angvel": float(np.linalg.norm(w)), "angvel_x": float(w[0]),
                "angvel_y": float(w[1]), "angvel_z": float(w[2]),
                "angvel_tilt": float(np.linalg.norm(w[:2])),
                "linvel": float(np.linalg.norm(v)), **dr}

    # ------------------------------------------------------------------
    def _done(self) -> bool:
        if self.bad_contact:
            return True
        t0 = self.t_first_foot if self.t_first_foot is not None else self.t_first_contact
        return t0 is not None and self.data.time - t0 >= self.settle_time - 1e-9

    def _task_info(self) -> dict:
        if self.impact is not None:
            v = self.impact["v"]
            vv, vh = float(max(-v[2], 0.0)), float(np.linalg.norm(v[:2]))
            tilt = math.degrees(self.impact["tilt"])
        else:
            vv = vh = float("nan")
            tilt = math.degrees(self._truth_tilt())
        landed_feet = self.first_contact in FEET
        safe = bool(landed_feet and not self.bad_contact and vv < self.lim_v and vh < self.lim_h
                    and tilt < self.lim_tilt)
        return {"safe": safe, "success": safe,
                "impact_v_vertical": vv, "impact_v_horizontal": vh,
                "impact_speed": float(math.hypot(vv, vh)),
                "tilt_at_impact_deg": float(tilt),
                "first_contact": self.first_contact,
                "bad_contact": bool(self.bad_contact),
                "any_bad_contact": bool(self.any_bad_contact),
                "t_first_contact": self.t_first_contact,
                "contact_time": self.t_first_contact,
                "failure_reasons": self._failure_reasons(landed_feet, vv, vh, tilt)}

    def _failure_reasons(self, landed_feet, vv, vh, tilt) -> list[str]:
        out = []
        if self.first_contact == "none":
            out.append("no_contact")
        elif not landed_feet:
            out.append("bad_contact" if self.bad_contact else "non_foot_first")
        if not vv < self.lim_v:
            out.append("v_vertical")
        if not vh < self.lim_h:
            out.append("v_horizontal")
        if not tilt < self.lim_tilt:
            out.append("tilt")
        return out

    def _reward(self, terminal: bool, info: dict) -> float:
        rw = self.rw
        r = 0.0
        # optional dense shaping (default 0): penalise gas spent this step
        k_shape = float(rw.get("shaping_impulse", 0.0))
        if k_shape:
            used = self.thrusters.used()
            tot = used["impulse_gas_pods"] + used["impulse_gas_jets"]
            r -= k_shape * (tot - getattr(self, "_used_prev", 0.0))
            self._used_prev = tot
        if not terminal:
            return r
        vv = info["impact_v_vertical"]
        vh = info["impact_v_horizontal"]
        speed2 = (vv * vv + vh * vh) if np.isfinite(vv) else 0.0
        r += float(rw.get("safe_bonus", 10.0)) * float(info["safe"])
        r -= float(rw.get("k1_impact_speed2", 0.0)) * speed2
        r -= float(rw.get("k2_tilt", 0.0)) * math.radians(info["tilt_at_impact_deg"])
        r -= float(rw.get("k3_impulse", 0.0)) * (info["impulse_gas_pods"] + info["impulse_gas_jets"])
        if info["bad_contact"] or info["first_contact"] == "none":
            r -= float(rw.get("bad_contact_penalty", 50.0))
        # optional safety-margin shaping (default 0): quadratic penalty once an impact
        # quantity comes within `margin_*` of its safe-landing limit (keeps an RL policy that
        # trades impact speed for gas away from the success cliff)
        k_m = float(rw.get("k_margin", 0.0))
        if k_m and np.isfinite(vv):
            mv, mh = float(rw.get("margin_v", 0.3)), float(rw.get("margin_h", 0.3))
            mt = float(rw.get("margin_tilt_deg", 5.0))
            ev = max(0.0, vv - (self.lim_v - mv)) / max(mv, 1e-9)
            eh = max(0.0, vh - (self.lim_h - mh)) / max(mh, 1e-9)
            et = max(0.0, info["tilt_at_impact_deg"] - (self.lim_tilt - mt)) / max(mt, 1e-9)
            r -= k_m * (ev * ev + eh * eh + et * et)
        return r

    def reset(self, *, seed=None, options=None):
        self._used_prev = 0.0
        return super().reset(seed=seed, options=options)
