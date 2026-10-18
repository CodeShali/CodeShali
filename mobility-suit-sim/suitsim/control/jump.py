"""Baseline (non-learning) boosted-jump controller for T2 (milestone M6).

Reads ONLY the estimator dict (``ESTIMATE_KEYS``), the estimator-derived thruster geometry
(``geometry_fn``) and mission knowledge from the task config (platform height / size /
nominal position relative to the start).  Never ground truth.  Gains from ``cfg['jump']``;
attitude gains from ``cfg['baseline']`` (shared with the drop-landing controller).

Phases
------
``stand``   estimator settles (nominal posture, no thrust).
``crouch``  posture servos ramp to ``jump.crouch_posture`` (joint targets).
``boost``   legs extend (``push_posture``) and the lift pods (+ fans) push ``boost_thrust``
            upward.  The boost ends when the *predicted ballistic apex* of the soles
            ``z_feet + v_z^2 / 2g`` reaches ``platform_height + apex_clearance`` — closed-loop,
            so thrust / mass randomisation only changes the burn time.
``flight``  ballistic: no lift thrust, attitude hold (thrust axis vertical), posture back to
            the nominal landing posture, horizontal guidance with the micro jets.
``descent`` braking profile on the soles' height ABOVE THE PLATFORM TOP (same law as the T1
            landing controller: ``v_ref = -sqrt(v_t^2 + 2 a (h - h_t))``); until the feet
            are over the platform the reference never descends below ``hold_height``.
``landed``  foot contact (or soles at ``cutoff_height``) -> thrust cut.

Horizontal guidance (all airborne phases): desired velocity toward the landing target
``v_des = sqrt(2 a_dec e)`` (capped), additionally capped so that the toes stay
``edge_margin_x`` behind the platform's near edge until the soles are ``edge_margin_z``
above its top (predicted ballistic time-to-clear).  ``a = k_v (v_des - v)`` is requested as
a horizontal force: micro jets in flight, jets + thrust-axis lean while thrusting.

Position / altitude: the estimator gives CoM height above the surface under the altimeter
(it re-acquires the platform top after a short reject window) and velocities, but no
position.  The controller integrates ``est['com_vel']`` from the known start (0, 0) and keeps
an absolute CoM altitude ``z`` (integrated ``v_z``) that is softly corrected by whichever
surface hypothesis — ``com_height`` (ground) or ``com_height + platform_height`` — is
consistent with it (gate ``alt_gate``).
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from suitsim.control.baseline import _DEFAULTS as _BASE_DEFAULTS, _quat_to_mat, _vec2
from suitsim.interfaces import ControlOutput, load_thruster_specs

_JUMP_DEFAULTS = {
    "mission_knows_platform": False,
    "settle_time": 0.10,
    "crouch_time": 0.35,
    "crouch_hold": 0.10,
    "crouch_posture": {},
    "push_posture": {},
    "leg_push": True,
    "flight_posture_delay": 0.08,
    "boost_thrust": 2900.0,
    "boost_ramp": 0.03,
    "apex_clearance": 0.40,
    "apex_drag_margin": 0.03,
    "target_offset_x": 0.0,
    "approach_decel": 1.5,
    "approach_speed_max": 2.0,
    "horizontal_kv": 4.0,
    "horizontal_accel_max": 4.0,
    "toe_offset": 0.22,
    "heel_offset": 0.12,
    "edge_margin_x": 0.15,
    "edge_margin_z": 0.15,
    "over_margin": 0.10,
    "target_speed": 0.9,
    "target_height": 0.25,
    "brake_decel_max": 8.0,
    "velocity_kp": 8.0,
    "hold_height": 0.35,
    "hold_kp": 3.0,
    "thrust_max": 3200.0,
    "tilt_cmd_max_deg": 10.0,
    "cutoff_height": 0.04,
    "alt_gate": 0.35,
    "alt_gain": 3.0,
    "boost_timeout": 0.8,
}


class BaselineJumpController:
    """Implements :class:`suitsim.interfaces.ControllerProtocol` for T2.

    ``debug['joint_targets']`` ({joint name: rad}) is applied to the posture servos by
    :class:`suitsim.envs.boosted_jump.BoostedJumpEnv`.
    """

    def __init__(self, cfg: dict, specs=None, geometry_fn: Callable | None = None):
        b = dict(_BASE_DEFAULTS)
        b.update(cfg.get("baseline", {}) or {})
        p = dict(_JUMP_DEFAULTS)
        p.update(cfg.get("jump", {}) or {})
        self.b, self.p = b, p
        self.g = float(cfg.get("sim", {}).get("gravity", 9.81))
        specs = specs if specs is not None else load_thruster_specs(cfg)
        self.pod_idx = np.array([i for i, s in enumerate(specs) if s.cls == "lift_pod"], int)
        self.geometry_fn = geometry_fn
        self.m = float(b["mass_nominal"])
        self.kp = _vec2(b["attitude_kp"])
        self.kd = _vec2(b["attitude_kd"])
        self.alpha = _vec2(b["attitude_alpha_max"])
        self.tmax = np.asarray(b["torque_max"], float)
        task = cfg.get("task", {}) or {}
        self.H = float(task.get("platform_height", 2.5))
        self.size = np.asarray(task.get("platform_size", [1.5, 1.5]), float)
        self.platform_xy_nominal = np.array([float(task.get("platform_distance", 1.5)), 0.0])
        self.platform_xy = self.platform_xy_nominal.copy()
        posture = (cfg.get("human", {}) or {}).get("posture", {}) or {}
        joints = list(((cfg.get("human", {}) or {}).get("joints", {}) or {}).get("dofs", {}))
        self.nominal = {j: math.radians(float(posture.get(j, 0.0))) for j in joints}

        def full(over):
            out = dict(self.nominal)
            out.update({k: math.radians(float(v)) for k, v in (over or {}).items()})
            return out

        self.crouch = full(p["crouch_posture"])
        self.push = full(p["push_posture"]) if p["leg_push"] else dict(self.crouch)
        self.sin_tilt = math.sin(math.radians(float(p["tilt_cmd_max_deg"])))
        self.reset()

    # ------------------------------------------------------------------------------------
    def set_mission(self, platform_xy=None) -> None:
        """Mission info from the env: the platform's position relative to the start (only
        used when ``jump.mission_knows_platform``; otherwise the nominal map is used)."""
        if platform_xy is not None and self.p["mission_knows_platform"]:
            self.platform_xy = np.asarray(platform_xy, float).copy()
        else:
            self.platform_xy = self.platform_xy_nominal.copy()

    def reset(self) -> None:
        self.phase = "stand"
        self.t0: float | None = None
        self.t_prev: float | None = None
        self.t_phase = 0.0
        self.x = np.zeros(2)
        self.z: float | None = None
        self.offset = 0.0           # com - feet height offset (kinematics)
        self.t_air: float | None = None
        self.braking = False
        self.last_geometry = None

    # ------------------------------------------------------------------------------------
    def thrust_axis(self, est: dict) -> np.ndarray:
        if self.geometry_fn is not None:
            r, d = self.geometry_fn(est)
            self.last_geometry = (r, d)
            z = d[self.pod_idx].mean(axis=0)
        else:
            z = _quat_to_mat(np.asarray(est["torso_quat"], float))[:, 2]
        return z / np.linalg.norm(z)

    def attitude_torque(self, est: dict, zp: np.ndarray, z_des: np.ndarray) -> np.ndarray:
        """Baseline (T1) saturation-aware attitude law: thrust axis -> z_des, yaw rate damped.
        Returns the world-frame torque."""
        R = _quat_to_mat(np.asarray(est["torso_quat"], float))
        w_b = np.asarray(est["torso_angvel"], float)
        ax = np.cross(zp, z_des)
        s = np.linalg.norm(ax)
        ang = math.atan2(s, float(zp @ z_des))
        e_b = R.T @ (ax / s * ang if s > 1e-9 else np.zeros(3))
        tau_b = np.zeros(3)
        for k in range(2):
            ek = e_b[k]
            w_lin = self.kp[k] / self.kd[k] * ek
            w_sq = math.copysign(math.sqrt(2.0 * self.alpha[k] * abs(ek)), ek)
            w_des = w_lin if abs(w_lin) < abs(w_sq) else w_sq
            tau_b[k] = self.kd[k] * (w_des - w_b[k])
        tau_b[2] = -float(self.b["yaw_rate_kd"]) * w_b[2]
        tau_b = np.clip(tau_b, -self.tmax, self.tmax)
        return R @ tau_b

    def v_ref(self, h: float) -> tuple[float, float]:
        p = self.p
        vt, ht, a = float(p["target_speed"]), float(p["target_height"]), float(p["brake_decel_max"])
        if h > ht:
            return -math.sqrt(vt * vt + 2.0 * a * (h - ht)), a
        return -vt, 0.0

    # ------------------------------------------------------------------------------------
    def _altitude(self, est: dict, dt: float) -> None:
        """Absolute CoM altitude (ground = 0) from integrated v_z + surface hypotheses."""
        p = self.p
        hc = float(est["com_height"])
        if self.z is None:
            self.z = hc
            return
        self.z += float(est["com_vel"][2]) * dt
        cands = (hc, hc + self.H)
        c = min(cands, key=lambda v: abs(v - self.z))
        if abs(c - self.z) < float(p["alt_gate"]):
            self.z += min(float(p["alt_gain"]) * dt, 1.0) * (c - self.z)

    def _horizontal(self, v: np.ndarray, feet: float, vz_pred: float) -> np.ndarray:
        """Desired horizontal acceleration (world xy)."""
        p = self.p
        tgt = self.platform_xy + np.array([float(p["target_offset_x"]), 0.0])
        e = tgt - self.x
        a_dec = float(p["approach_decel"])
        vmax = float(p["approach_speed_max"])
        v_des = np.array([math.copysign(min(math.sqrt(2.0 * a_dec * abs(ek)), vmax), ek)
                          for ek in e])
        # edge clearance: toes behind the near edge until the soles clear the top
        z_clear = self.H + float(p["edge_margin_z"])
        if feet < z_clear:
            edge = self.platform_xy[0] - self.size[0] / 2 - float(p["edge_margin_x"])
            room = edge - (self.x[0] + float(p["toe_offset"]))
            disc = vz_pred * vz_pred - 2.0 * self.g * (z_clear - feet)
            if disc <= 0.0 or vz_pred <= 0.0:
                cap = 0.0 if room <= 0 else math.sqrt(2.0 * a_dec * room)
            else:
                t_clear = (vz_pred - math.sqrt(disc)) / self.g
                cap = max(room, 0.0) / max(t_clear, 1e-3)
            v_des[0] = min(v_des[0], cap)
        a = float(p["horizontal_kv"]) * (v_des - v[:2])
        n = np.linalg.norm(a)
        amax = float(p["horizontal_accel_max"])
        if n > amax:
            a *= amax / n
        return a

    def _over_platform(self) -> bool:
        p = self.p
        half = self.size / 2
        rel = self.x - self.platform_xy
        m = float(p["over_margin"])
        return bool(rel[0] - float(p["heel_offset"]) > -half[0] + m
                    and rel[0] + float(p["toe_offset"]) < half[0] - m
                    and abs(rel[1]) < half[1] - m)

    # ------------------------------------------------------------------------------------
    def compute(self, est: dict, sensors: dict | None = None) -> ControlOutput:
        p, g, m = self.p, self.g, self.m
        t = float(est["t"])
        if self.t0 is None:
            self.t0 = t
            self.t_phase = t
        dt = 0.0 if self.t_prev is None else max(t - self.t_prev, 0.0)
        self.t_prev = t
        v = np.asarray(est["com_vel"], float)
        contact = bool(np.any(est["foot_contact"]))
        self._altitude(est, dt)
        self.offset = float(est["com_height"]) - float(est["feet_height"])
        feet = self.z - self.offset                    # absolute sole height
        if self.phase not in ("stand", "crouch"):
            self.x += v[:2] * dt
        zero = np.zeros(6)
        dbg = {"z_abs": self.z, "feet_abs": feet, "x": self.x.tolist()}
        tp = t - self.t_phase

        if self.phase == "stand":
            if tp >= float(p["settle_time"]):
                self._goto("crouch", t)
            return ControlOutput(wrench=zero, phase="stand",
                                 debug={**dbg, "joint_targets": dict(self.nominal)})
        if self.phase == "crouch":
            s = min(tp / max(float(p["crouch_time"]), 1e-6), 1.0)
            jt = {k: (1 - s) * self.nominal[k] + s * self.crouch[k] for k in self.nominal}
            if tp >= float(p["crouch_time"]) + float(p["crouch_hold"]):
                self._goto("boost", t)
            return ControlOutput(wrench=zero, phase="crouch", debug={**dbg, "joint_targets": jt})

        zp = self.thrust_axis(est)
        if self.phase in ("boost", "flight", "descent") and self.t_air is None \
                and not contact and feet > 0.03:
            self.t_air = t
        airborne = self.t_air is not None
        if airborne and t - self.t_air >= float(p["flight_posture_delay"]):
            jt = dict(self.nominal)
        elif self.phase == "boost":
            jt = dict(self.push)
        else:
            jt = dict(self.nominal)
        dbg["joint_targets"] = jt

        if self.phase == "landed":
            return ControlOutput(wrench=zero, phase="landed", debug=dbg)

        apex = feet + max(v[2], 0.0) ** 2 / (2.0 * g)
        apex_tgt = self.H + float(p["apex_clearance"]) + float(p["apex_drag_margin"])
        if self.phase == "boost":
            if apex >= apex_tgt or tp > float(p["boost_timeout"]):
                self._goto("flight", t)
        if self.phase == "flight" and v[2] < 0.0:
            self._goto("descent", t)
        h_rel = feet - self.H
        over = self._over_platform()
        if self.phase == "descent" and over and (contact or h_rel < float(p["cutoff_height"])):
            self._goto("landed", t)
            return ControlOutput(wrench=zero, phase="landed", debug=dbg)

        # predicted vertical speed at the end of the boost (for the edge-clearance cap)
        vz_pred = math.sqrt(max(2.0 * g * (apex_tgt - feet), 0.0)) if self.phase == "boost" \
            else v[2]
        a_xy = self._horizontal(v, feet, vz_pred) if airborne or self.phase != "boost" \
            else np.zeros(2)

        if self.phase == "boost":
            ramp = min(tp / max(float(p["boost_ramp"]), 1e-6), 1.0)
            Fz = ramp * float(p["boost_thrust"])
            z_des = np.array([0.0, 0.0, 1.0])
            F = np.array([m * a_xy[0], m * a_xy[1], Fz])
        elif self.phase == "flight":
            z_des = np.array([0.0, 0.0, 1.0])
            F = np.array([m * a_xy[0], m * a_xy[1], 0.0])
            Fz = 0.0
        else:   # descent
            vr, a_plan = self.v_ref(h_rel)
            if not over:
                vr = max(vr, float(p["hold_kp"]) * (float(p["hold_height"]) - h_rel))
            on_profile = v[2] <= vr
            if on_profile:
                self.braking = True
            if self.braking:
                a_ff = a_plan if on_profile else 0.0
                a_z = a_ff + float(p["velocity_kp"]) * (vr - v[2])
                Fz = float(np.clip(m * (g + a_z), 0.0, float(p["thrust_max"])))
            else:
                Fz = 0.0
            F = np.array([m * a_xy[0], m * a_xy[1], Fz])
            dbg["v_ref"] = vr
            if Fz > 1e-6:
                lean = F[:2] / Fz
                n = np.linalg.norm(lean)
                if n > self.sin_tilt:
                    lean *= self.sin_tilt / n
                z_des = np.array([lean[0], lean[1], 1.0])
                z_des /= np.linalg.norm(z_des)
            else:
                z_des = np.array([0.0, 0.0, 1.0])
        tau = self.attitude_torque(est, zp, z_des)
        dbg.update({"Fz": Fz, "a_xy": a_xy.tolist(), "apex_pred": apex, "h_rel": h_rel,
                    "over": over})
        return ControlOutput(wrench=np.concatenate([F, tau]), phase=self.phase, debug=dbg)

    def _goto(self, phase: str, t: float) -> None:
        self.phase = phase
        self.t_phase = t
