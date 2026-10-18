"""Standing-balance / push-recovery controller (T3, milestone M6 part 2).

Reads ONLY estimator outputs (``est``: attitude from gyro + quasi-static gravity
correction, joint angles / rates) and raw sensor readings (foot pressure, gyro); never
ground truth.  Every gain / limit comes from ``cfg['balance']``.

Model: linear inverted pendulum on the support polygon
------------------------------------------------------
With the feet flat and the (servo-held) body stiff, the horizontal CoM dynamics are::

    c'' = w0^2 (c - p) + u,        u = (F_h + T_feet) / m,        w0 = sqrt(g / h)

``p`` = centre of pressure (bounded by the support polygon), ``h`` = CoM height above the
soles and ``u`` the thruster contribution: the thruster wrench about the CoM ``[F, tau]``
enters only through its MOMENT ABOUT THE FEET, i.e. ``u_x = (F_x + tau_y / h) / m`` and
``u_y = (F_y - tau_x / h) / m``.  So a lift-pod pair used differentially (a roll couple with
collective lift) counts as lateral authority just like a side jet:
``pod_l`` (y = +0.2 m) at force f gives ``u_y ~ -0.19 f / m``.

Capture-point (ICP) law
-----------------------
``xi = c + v / w0`` (relative to the support centre, in the support frame).  Desired
``xi' = -k_xi xi`` needs the CMP ``p_need = xi (1 + k_xi / w0)``; the feet supply it as long
as ``p_need`` stays inside the support polygon shrunk by ``cop_margin``; the remainder is
requested from the thrusters, ``u = w0^2 (clip(p_need) - p_need) * gain_u``.

Estimation of ``c``, ``v`` (relative to the feet)
------------------------------------------------
``c - sole_mid`` from the estimator's kinematics (estimated torso attitude + measured joint
angles); its rate analytically from the torso gyro + joint-velocity encoders (one
perturbed kinematics evaluation) — low noise, no differentiation, no drift; an optional
blend with the IMU-integrated ``est['com_vel']``.  The support frame's x axis is the mean
horizontal foot axis; loaded feet come from the foot-pressure sensors.

Allocation
----------
Own 6-row box QP (re-using :class:`~suitsim.control.fast_allocation.FastAllocator`'s solver):
rows ``[F_x + tau_y/h, F_y - tau_x/h, F_z, tau_z, tau_x, tau_y]`` with targets
``[m u_x, m u_y, F_z,des, -k_yaw w_z, tau_att]`` and weights from the config — collective
lift is tolerated (soft) up to ``lift_max``, torso attitude is weakly regulated.
Nominal (YAML) ``f_max`` bounds are used, zeroed for empty budgets (``est['budgets']``).

Posture strategy (optional, ``hip_strategy_gain``)
--------------------------------------------------
Posture-servo offsets (returned in ``debug['posture_offset']``, applied by the env): hip
ab/adduction + lumbar roll in the lateral plane and hip/ankle flexion in the sagittal plane,
proportional to the ICP excess.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from suitsim.control.fast_allocation import FastAllocator
from suitsim.interfaces import FEET, ControlOutput, load_thruster_specs

_DEFAULTS = {
    "mass_nominal": 100.0,          # [kg]
    "com_height_nominal": 1.0,      # [m] fallback CoM height above the soles
    "foot_half_length": None,       # [m] default: human.geometry.foot.size[0]
    "foot_half_width": None,        # [m] default: human.geometry.foot.size[1]
    "cop_margin": [0.07, 0.07],     # [m] (x, y) support polygon shrink for the feet's share
    "k_xi": 3.0,                    # [1/s] capture-point convergence rate
    "gain_u": [1.5, 1.5],           # thruster accel multiplier (x, y)
    "u_max": [6.0, 6.0],            # [m/s^2] cap of the requested thruster accel
    "deadband_u": 0.05,             # [m/s^2] below this no thrust is requested
    "xi_ref": [0.0, 0.0],           # [m] capture-point target relative to the support centre
    "vel_blend": 0.0,               # weight of est['com_vel'] vs kinematic velocity
    "vel_filter_hz": 40.0,          # first-order LPF of the kinematic CoM velocity
    "pos_filter_hz": 60.0,
    "lift_max": 600.0,              # [N] collective lift tolerated
    "w_u": 1.0,                     # row weights
    "w_fz": 0.25,
    "w_yaw": 1.0,
    "w_att": 0.15,
    "yaw_rate_kd": 15.0,            # [N m s/rad]
    "att_kp": 300.0,                # [N m/rad] weak torso attitude PD (target: initial tilt)
    "att_kd": 60.0,
    "lambda_scale": 1.0,
    "load_frac": 0.15,              # foot counted as support if it carries > this share
    "hip_strategy_gain": [0.0, 0.0],  # [rad per m of ICP excess] (sagittal, lateral)
    "hip_strategy_max_deg": 20.0,
    "gravity": None,
}


def _vec2(x) -> np.ndarray:
    a = np.atleast_1d(np.asarray(x, dtype=float))
    return np.repeat(a, 2) if a.size == 1 else a[:2].copy()


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    m = np.empty(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


class BalanceController:
    """Implements :class:`suitsim.interfaces.ControllerProtocol` (``compute(est, sensors)``).

    Parameters
    ----------
    cfg : merged config (gains in ``cfg['balance']``).
    specs : thruster specs (default from cfg).
    estimator : the :class:`~suitsim.estimation.StateEstimator` whose ``est`` is passed in
        (its kinematics model and ``thruster_geometry`` are used — sensor-derived only).
    sensors_fn : optional ``() -> sensors dict`` (latest reading, for foot pressure);
        ``compute``'s ``sensors`` argument wins when given.
    """

    def __init__(self, cfg: dict, specs=None, estimator=None, sensors_fn=None):
        p = dict(_DEFAULTS)
        p.update(cfg.get("balance", {}) or {})
        self.p = p
        self.cfg = cfg
        self.g = float(p["gravity"] if p["gravity"] is not None
                       else cfg.get("sim", {}).get("gravity", 9.81))
        self.specs = specs if specs is not None else load_thruster_specs(cfg)
        self.n = len(self.specs)
        self.est_obj = estimator
        self.sensors_fn = sensors_fn
        self.m = float(p["mass_nominal"])
        foot = cfg["human"]["geometry"]["foot"]["size"]
        self.foot_hl = float(p["foot_half_length"] or foot[0])
        self.foot_hw = float(p["foot_half_width"] or foot[1])
        self.margin = _vec2(p["cop_margin"])
        self.gain_u = _vec2(p["gain_u"])
        self.umax = _vec2(p["u_max"])
        self.xi_ref = _vec2(p["xi_ref"])
        self.hip_gain = _vec2(p["hip_strategy_gain"])
        self.alloc = FastAllocator(self.specs, cfg)
        self.scale = np.array([s.f_max for s in self.specs])
        self.lam = self.alloc.lam.copy()
        self.budget_of = [s.budget for s in self.specs]
        self.is_pod = np.array([s.cls == "lift_pod" for s in self.specs])
        if estimator is not None:
            kin = estimator.kin
            m = kin.model
            self.foot_bids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f) for f in FEET]
            # hinge order of the estimator (== sensor joint order)
            hinge = np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
            names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in hinge]
            self.jidx = {n: k for k, n in enumerate(names)}
            self.n_joints = len(names)
        self.reset()

    # ------------------------------------------------------------------------------------
    def reset(self) -> None:
        self.t_prev = None
        self.c_f = None
        self.v_f = None
        self.att_ref = None
        self.alloc.reset_warm_start()
        self.last = {}

    # ------------------------------------------------------------------------------------
    def _feet_state(self, est: dict):
        """Kinematic CoM / soles (world-aligned, rel. root) + their rate from gyro/encoders."""
        e = self.est_obj
        kin_m = e.kin
        q = np.asarray(est["torso_quat"], float)
        jq = np.asarray(est["joint_pos"], float)
        jv = np.asarray(est["joint_vel"], float)
        w_b = np.asarray(est["torso_angvel"], float)
        k0 = kin_m.compute(q, jq)
        R_root = k0.R_root
        foot_x = [R_root @ kin_m.data.xmat[b].reshape(3, 3)[:, 0] for b in self.foot_bids]
        eps = 0.01
        q2 = q.copy()
        mujoco.mju_quatIntegrate(q2, w_b, eps)
        k1 = kin_m.compute(q2, jq + jv * eps)
        rel0 = k0.com[None, :] - k0.sole_pos           # (2,3) CoM rel. each sole
        rel1 = k1.com[None, :] - k1.sole_pos
        vrel = (rel1 - rel0) / eps
        return rel0, vrel, np.array(foot_x)

    # ------------------------------------------------------------------------------------
    def compute(self, est: dict, sensors: dict | None = None) -> ControlOutput:
        p, g, m = self.p, self.g, self.m
        t = float(est["t"])
        dt = 0.0 if self.t_prev is None else max(t - self.t_prev, 0.0)
        self.t_prev = t
        if sensors is None and self.sensors_fn is not None:
            sensors = self.sensors_fn()
        fp = (np.asarray(sensors["foot_pressure"], float) if sensors is not None
              else np.asarray(est["foot_contact"], float))
        tot = float(fp.sum())
        loaded = fp > float(p["load_frac"]) * max(tot, 1e-6)
        if not loaded.any():
            loaded = np.ones(2, bool)

        rel, vrel, foot_x = self._feet_state(est)
        # support frame: mean horizontal foot axis
        fx = foot_x[:, :2].sum(axis=0)
        nfx = np.linalg.norm(fx)
        ex = fx / nfx if nfx > 1e-6 else np.array([1.0, 0.0])
        ey = np.array([-ex[1], ex[0]])
        Rs = np.vstack([ex, ey])                     # support <- world (2x2)
        # support centre = mid of loaded soles; CoM relative to it
        w = loaded.astype(float) / loaded.sum()
        c_w = w @ rel                                # (3,) CoM rel. support centre (world)
        v_w = w @ vrel
        h = float(c_w[2]) if c_w[2] > 0.3 else float(p["com_height_nominal"])
        if p["vel_blend"]:
            b = float(p["vel_blend"])
            v_w[:2] = (1 - b) * v_w[:2] + b * np.asarray(est["com_vel"], float)[:2]
        c = Rs @ c_w[:2]
        v = Rs @ v_w[:2]
        # low-pass filters
        if self.c_f is None or dt <= 0:
            self.c_f, self.v_f = c.copy(), v.copy()
        else:
            a_c = 1.0 - math.exp(-2 * math.pi * float(p["pos_filter_hz"]) * dt)
            a_v = 1.0 - math.exp(-2 * math.pi * float(p["vel_filter_hz"]) * dt)
            self.c_f += a_c * (c - self.c_f)
            self.v_f += a_v * (v - self.v_f)
        c, v = self.c_f, self.v_f

        # support polygon (support frame, relative to the support centre)
        sole_s = np.array([Rs @ (-(r[:2] - c_w[:2])) for r in rel])  # sole centres
        # (sole_k - centre) = c_w - rel_k
        sole_s = np.array([Rs @ (c_w[:2] - r[:2]) for r in rel])
        lo = np.min(sole_s[loaded] - [self.foot_hl, self.foot_hw], axis=0)
        hi = np.max(sole_s[loaded] + [self.foot_hl, self.foot_hw], axis=0)
        lo_m = np.minimum(lo + self.margin, -0.005)
        hi_m = np.maximum(hi - self.margin, 0.005)

        w0 = math.sqrt(g / h)
        xi = c + v / w0
        k = float(p["k_xi"])
        p_need = self.xi_ref + (xi - self.xi_ref) * (1.0 + k / w0)
        p_ok = np.clip(p_need, lo_m, hi_m)
        u_s = w0 * w0 * (p_ok - p_need) * self.gain_u
        u_s = np.clip(u_s, -self.umax, self.umax)
        if np.linalg.norm(u_s) < float(p["deadband_u"]):
            u_s = np.zeros(2)
        u_w = Rs.T @ u_s                              # world horizontal accel request

        # torso attitude (weak) + yaw damping
        R = _quat_to_mat(est["torso_quat"])
        zt = R[:, 2]
        if self.att_ref is None:
            self.att_ref = zt.copy()
        ax = np.cross(zt, self.att_ref)
        wb = np.asarray(est["torso_angvel"], float)
        w_world = R @ wb
        tau_att = float(p["att_kp"]) * ax[:2] - float(p["att_kd"]) * w_world[:2]
        tau_z = -float(p["yaw_rate_kd"]) * w_world[2]

        # posture strategy
        posture = None
        if np.any(self.hip_gain):
            posture = self._posture(p_ok - p_need, Rs)

        need = np.linalg.norm(u_s) > 0
        if not need:
            u_cmd = np.zeros(self.n)
            fz_des = 0.0
        else:
            u_cmd, fz_des = self._allocate(est, u_w, h, tau_att, tau_z)
        dbg = {"xi_x": float(xi[0]), "xi_y": float(xi[1]), "c_x": float(c[0]),
               "c_y": float(c[1]), "v_x": float(v[0]), "v_y": float(v[1]),
               "u_x": float(u_s[0]), "u_y": float(u_s[1]), "h": h,
               "loaded": int(loaded[0]) + 2 * int(loaded[1])}
        if posture is not None:
            dbg["posture_offset"] = posture
        self.last = dbg
        wrench = np.array([m * u_w[0], m * u_w[1], fz_des, 0.0, 0.0, 0.0])
        return ControlOutput(wrench=wrench, thrust_cmd=u_cmd,
                             phase="balance" if need else "stand", debug=dbg)

    # ------------------------------------------------------------------------------------
    def _allocate(self, est, u_w, h, tau_att, tau_z):
        p, m = self.p, self.m
        r, d = self.est_obj.thruster_geometry(est)
        A = self.alloc.matrix(r, d)
        rows = np.vstack([A[0] + A[4] / h, A[1] - A[3] / h, A[2], A[5], A[3], A[4]])
        tgt = np.array([m * u_w[0], m * u_w[1], 0.0, tau_z, tau_att[0], tau_att[1]])
        wu, wz, wy, wa = (float(p["w_u"]), float(p["w_fz"]), float(p["w_yaw"]),
                          float(p["w_att"]))
        Wr = np.array([wu, wu, wz, wy / 0.3, wa / 0.3, wa / 0.3])
        bud = est.get("budgets") or {}
        ub_f = self.scale.copy()
        for i, b in enumerate(self.budget_of):
            if bud.get(b, 1.0) <= 0.0:
                ub_f[i] = 0.0
        act = ub_f > 1e-9
        u = np.zeros(self.n)
        if not act.any():
            return u, 0.0
        s = self.scale[act]
        M = (Wr[:, None] * rows[:, act]) * s[None, :]
        b = Wr * tgt
        ub = ub_f[act] / s
        v, _ = self.alloc._solve(M, b, np.sqrt(float(p["lambda_scale"]) * self.lam[act]), ub)
        u[act] = v * s
        # soft lift cap: scale the pods down if collective lift exceeds lift_max
        fz = float(A[2] @ u)
        lmax = float(p["lift_max"])
        if fz > lmax:
            pods = self.is_pod
            fz_p = float(A[2, pods] @ u[pods])
            if fz_p > 1e-6:
                u[pods] *= max(0.0, (lmax - (fz - fz_p)) / fz_p)
        return u, float(A[2] @ u)

    # ------------------------------------------------------------------------------------
    def _posture(self, excess_s, Rs):
        """Posture-servo offsets [rad] (hinge order) from the ICP excess (support frame)."""
        lim = math.radians(float(self.p["hip_strategy_max_deg"]))
        off = np.zeros(self.n_joints)
        ex, ey = excess_s * self.hip_gain           # >0 : needs push toward +x / +y
        ex = float(np.clip(ex, -lim, lim))
        ey = float(np.clip(ey, -lim, lim))
        J = self.jidx
        # lateral: lean the trunk toward the fall side (hip strategy): falling to -y needs
        # +y accel (ey > 0) -> roll the trunk to -y quickly (reaction pushes CoM to +y)
        if "lumbar_roll" in J:
            off[J["lumbar_roll"]] = ey
        if "hip_l_abd" in J:
            off[J["hip_l_abd"]] = -ey
            off[J["hip_r_abd"]] = ey
        # sagittal: hip flexion (bend forward when falling backward needs +x accel...)
        if "hip_l_flex" in J:
            off[J["hip_l_flex"]] = ex
            off[J["hip_r_flex"]] = ex
        return off
