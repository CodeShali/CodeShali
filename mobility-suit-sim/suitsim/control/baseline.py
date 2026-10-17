"""Baseline (non-learning) landing controller, spec 5.2 (milestone M4).

Reads ONLY the estimator dict (``ESTIMATE_KEYS``) plus the thrust-axis geometry derived from
it (``geometry_fn(est) -> (r, d)``, i.e. ``StateEstimator.thruster_geometry``).  Never ground
truth.  Every gain / limit comes from ``cfg['baseline']``.

Options kept for experiments, OFF by default because they tested worse on T1:
``yaw_align`` (yaw so the demand is on the strong pitch axis), ``torque_feasible_clip``,
``jet_horizontal_gain`` (jets burn their budget), ``assist_horizontal_accel_max`` (100 = off).

Structure (quadrotor-style, the lift pods are the "rotors")
----------------------------------------------------------
1. **Fall detection** — nothing is commanded until ``est['free_fall']`` latches.
2. **Vertical channel** — braking profile on the *feet height* ``h``::

       v_ref(h) = -sqrt(v_t^2 + 2 a_plan (h - h_t))   for h > h_t,   v_ref = -v_t below

   Braking starts when the measured descent speed reaches the profile (the profile is an
   "intersection" test, so the gas is spent as late as possible); the command is
   ``a_z = a_ff + k_v (v_ref - v_z) + k_i * int(v_ref - v_z)`` with ``a_ff = a_plan`` while
   on the profile.  Feed-forward mass is the NOMINAL mass (the true mass is randomised
   +-10 %); the integrator absorbs mass / thrust-scale errors.
3. **Horizontal channel** — ``a_xy = -k_h v_xy`` (clamped), realised by tilting the thrust
   axis (mean lift-pod direction) towards ``F_des = m (a + g z)``, limited to
   ``tilt_cmd_max_deg`` from vertical.
4. **Attitude** — drives the thrust axis onto the desired axis (tilt error only, yaw angle
   ignored) and damps yaw rate.  Saturation-aware: the commanded rate is the minimum of the
   linear PD rate ``(kp/kd) e`` and the time-optimal ``sqrt(2 alpha_max e)`` (per body axis),
   torque ``tau = kd (w_des - w)`` clipped per axis.  Roll torque from the micro jets alone
   is small (~+-50 N*m), so when the roll demand exceeds ``roll_jet_authority`` a minimum
   pod thrust ``T >= (|tau_x| - roll_jet_authority)/pod_arm`` is requested so the allocator
   can use the pods differentially.
5. **Thrust** — magnitude from a weighted least-squares fit of ``T z_p`` to ``F_des``
   (vertical vs horizontal error weighted by ``thrust_horizontal_weight``), so a tilted
   body does not convert braking into horizontal speed (and thrust is boosted when the
   tilt points the right way); faded out between ``thrust_tilt_full_deg`` and
   ``thrust_tilt_cut_deg``.  Roll-assist thrust is capped so that its horizontal
   acceleration stays below ``assist_horizontal_accel_max`` unless it helps.  The force
   handed to the allocator is ``T * z_p`` — always realisable by the pods/fans.
   With ``horizontal_in_attitude`` the desired axis also leans against ``v_xy`` before
   braking starts, so the (unavoidable) roll-assist thrust helps instead of only adding
   horizontal speed (tuning: 98.8 % -> 99.8 % on a 400-episode T1 batch).
   Fall detection: ``est['free_fall']`` OR estimated descent faster than
   ``fall_vz_thresh`` without foot contact (the estimator's torso-IMU free-fall test can
   latch late for spinning bodies).
6. **Touchdown** — thrust latched off at foot contact or ``feet_height < cutoff_height``;
   airbags deploy when the predicted impact speed ``sqrt(v_z^2 + 2 g h)`` (evaluated below
   ``airbag_check_height``) exceeds ``airbag_speed`` or the tilt near the ground exceeds
   ``airbag_tilt_deg``.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from suitsim.interfaces import ControlOutput, load_thruster_specs

_DEFAULTS = {
    "mass_nominal": 100.0,           # [kg] feed-forward mass (true mass is randomised)
    "inertia_nominal": [19.0, 18.0, 1.75],   # [kg m^2] body-axis inertia about CoM
    "attitude_kp": [2600.0, 2400.0],   # [N m/rad] (roll, pitch) scalar or [kx, ky]
    "attitude_kd": [380.0, 360.0],     # [N m s/rad]
    "attitude_alpha_max": [8.0, 8.0],  # [rad/s^2] planning ang. accel for sqrt law
    "torque_max": [300.0, 250.0, 70.0],  # [N m] clip per body axis
    "yaw_rate_kd": 15.0,             # [N m s/rad]
    "roll_jet_authority": 40.0,      # [N m] roll torque the jets give without pods
    "pod_arm": 0.2,                  # [m] lateral pod offset from CoM
    "assist_thrust_max": 1200.0,     # [N] cap of attitude-assist pod thrust
    "target_speed": 1.2,
    "target_height": 0.3,
    "brake_decel_max": 9.0,          # [m/s^2] planning deceleration of the profile
    "velocity_kp": 8.0,
    "velocity_ki": 0.0,              # [1/s^2]
    "integral_max": 3.0,             # [m/s^2] clamp of the integral term
    "accel_cmd_max": 20.0,           # [m/s^2] clamp of commanded vertical accel (above g)
    "horizontal_kp": 2.0,
    "horizontal_accel_max": 3.0,     # [m/s^2]
    "tilt_cmd_max_deg": 10.0,
    "thrust_tilt_full_deg": 25.0,    # full vertical-compensated thrust below this tilt
    "thrust_tilt_cut_deg": 70.0,     # no braking thrust beyond this tilt (linear between)
    "thrust_max": 3000.0,            # [N] cap of the requested thrust magnitude
    "thrust_horizontal_weight": 1.0,  # w_h/w_z of the thrust-magnitude LS
    "assist_horizontal_accel_max": 100.0,  # [m/s^2] horizontal accel tolerated from assist
                                     # (100 = off; a tight cap slows uprighting)
    "horizontal_in_attitude": True,  # tilt the desired axis against v_xy also when no
                                      # braking thrust is commanded (assist thrust helps)
    "yaw_align": False,              # steer yaw so attitude demand is about the pitch axis
    "yaw_align_min_deg": 8.0,        # active while thrust-axis error exceeds this
    "yaw_align_kp": 8.0,             # [1/s]
    "yaw_align_alpha": 20.0,         # [rad/s^2] sqrt-law planning accel
    "yaw_align_kd": 20.0,            # [N m s/rad]
    "torque_feasible_clip": False,   # clip roll torque to jets + pod differential at T
    "jet_horizontal_gain": 0.0,      # fraction of the unrealised horizontal force sent to jets
    "jet_horizontal_force_max": 300.0,  # [N]
    "airbag_speed": 2.5,
    "airbag_check_height": 0.5,      # [m] feet height below which impact is predicted
    "airbag_tilt_deg": 45.0,
    "cutoff_height": 0.05,
    "fall_vz_thresh": 0.4,           # [m/s] fallback fall detection: descending faster
                                     # than this without foot contact (spinning bodies can
                                     # keep the torso IMU above the free-fall threshold)
    "near_ground_height": 0.6,       # [m] below this the commanded tilt limit shrinks
    "tilt_cmd_max_near_deg": 10.0,
    "gravity": None,                 # default: cfg.sim.gravity
}


def _vec2(x) -> np.ndarray:
    a = np.atleast_1d(np.asarray(x, dtype=float))
    return np.repeat(a, 2) if a.size == 1 else a[:2]


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class BaselineLandingController:
    """Implements :class:`suitsim.interfaces.ControllerProtocol`.

    Parameters
    ----------
    cfg : full (merged) config; gains from ``cfg['baseline']``.
    specs : thruster specs (default ``load_thruster_specs(cfg)``) — used for the lift-pod
        indices (thrust axis = mean pod direction).
    geometry_fn : ``est -> (r, d)`` world-aligned thruster geometry (estimator output).  If
        None the thrust axis is approximated by the torso z axis.
    """

    def __init__(self, cfg: dict, specs=None, geometry_fn: Callable | None = None):
        p = dict(_DEFAULTS)
        p.update(cfg.get("baseline", {}) or {})
        self.p = p
        self.g = float(p["gravity"] if p["gravity"] is not None
                       else cfg.get("sim", {}).get("gravity", 9.81))
        specs = specs if specs is not None else load_thruster_specs(cfg)
        self.pod_idx = np.array([i for i, s in enumerate(specs) if s.cls == "lift_pod"], int)
        self.geometry_fn = geometry_fn
        self.m = float(p["mass_nominal"])
        self.I = np.asarray(p["inertia_nominal"], float)
        self.kp = _vec2(p["attitude_kp"])
        self.kd = _vec2(p["attitude_kd"])
        self.alpha = _vec2(p["attitude_alpha_max"])
        self.tmax = np.asarray(p["torque_max"], float)
        self.cos_cmd = math.cos(math.radians(float(p["tilt_cmd_max_deg"])))
        self.sin_cmd = math.sin(math.radians(float(p["tilt_cmd_max_deg"])))
        self.cos_near = math.cos(math.radians(float(p["tilt_cmd_max_near_deg"])))
        self.sin_near = math.sin(math.radians(float(p["tilt_cmd_max_near_deg"])))
        self.cos_full = math.cos(math.radians(float(p["thrust_tilt_full_deg"])))
        self.tilt_full = math.radians(float(p["thrust_tilt_full_deg"]))
        self.tilt_cut = math.radians(float(p["thrust_tilt_cut_deg"]))
        self.reset()

    # ------------------------------------------------------------------------------------
    def reset(self) -> None:
        self.landed = False
        self.falling = False
        self.airbags = False
        self.braking = False
        self.integ = 0.0
        self.t_prev: float | None = None
        self.last_geometry = None

    # ------------------------------------------------------------------------------------
    def v_ref(self, h: float) -> tuple[float, float]:
        """(reference vertical velocity, profile feed-forward decel) at feet height h."""
        p = self.p
        vt, ht, a = float(p["target_speed"]), float(p["target_height"]), float(p["brake_decel_max"])
        if h > ht:
            return -math.sqrt(vt * vt + 2.0 * a * (h - ht)), a
        return -vt, 0.0

    def thrust_axis(self, est: dict) -> np.ndarray:
        if self.geometry_fn is not None:
            r, d = self.geometry_fn(est)
            self.last_geometry = (r, d)
            z = d[self.pod_idx].mean(axis=0)
        else:
            z = _quat_to_mat(np.asarray(est["torso_quat"], float))[:, 2]
        return z / np.linalg.norm(z)

    # ------------------------------------------------------------------------------------
    def compute(self, est: dict, sensors: dict | None = None) -> ControlOutput:
        p, g, m = self.p, self.g, self.m
        t = float(est["t"])
        dt = 0.0 if self.t_prev is None else max(t - self.t_prev, 0.0)
        self.t_prev = t
        zero = np.zeros(6)
        h = float(est["feet_height"])
        v = np.asarray(est["com_vel"], float)
        contact = bool(np.any(est["foot_contact"]))
        if not self.falling:
            self.falling = bool(est["free_fall"]) or (
                v[2] < -float(p["fall_vz_thresh"]) and not contact)
        if not self.falling and not self.landed:
            return ControlOutput(wrench=zero, phase="idle")

        tilt = float(est["tilt"])

        # ---- touchdown / airbags --------------------------------------------------------
        if not self.landed and (contact or h < float(p["cutoff_height"])):
            self.landed = True
        if not self.airbags and h < float(p["airbag_check_height"]):
            # conservative: kinetic energy in excess of the braking profile is not removed;
            # thrust tracks the profile down to cutoff_height, then free fall
            vr_h, _ = self.v_ref(max(h, 0.0))
            vt = float(p["target_speed"])
            h_ff = min(max(h, 0.0), float(p["cutoff_height"]))
            v_pred = math.sqrt(max(v[2] * v[2] - vr_h * vr_h, 0.0) + vt * vt + 2.0 * g * h_ff) \
                if v[2] < 0 else 0.0
            if v_pred > float(p["airbag_speed"]) or tilt > math.radians(float(p["airbag_tilt_deg"])):
                self.airbags = True
        if self.landed:
            return ControlOutput(wrench=zero, deploy_airbags=self.airbags, phase="landed")

        # ---- vertical channel ----------------------------------------------------------
        vr, a_plan = self.v_ref(h)
        err = vr - v[2]
        on_profile = v[2] <= vr          # descending at least as fast as the profile
        if on_profile:
            self.braking = True
        a_ff = a_plan if on_profile else 0.0
        if self.braking:
            self.integ = float(np.clip(self.integ + float(p["velocity_ki"]) * err * dt,
                                       -p["integral_max"], p["integral_max"]))
        a_z = a_ff + float(p["velocity_kp"]) * err + self.integ
        a_z = min(a_z, float(p["accel_cmd_max"]))

        # ---- horizontal channel + desired thrust axis ------------------------------------
        a_xy = -float(p["horizontal_kp"]) * v[:2]
        n = np.linalg.norm(a_xy)
        amax = float(p["horizontal_accel_max"])
        if n > amax:
            a_xy *= amax / n
        Fz_des = m * (g + a_z)
        F_des = np.array([m * a_xy[0], m * a_xy[1], Fz_des])
        Fz_dir = max(Fz_des, m * g) if p["horizontal_in_attitude"] else Fz_des
        if Fz_dir > 1e-6:
            F_dir = np.array([F_des[0], F_des[1], Fz_dir])
            z_des = F_dir / np.linalg.norm(F_dir)
            near = h < float(p["near_ground_height"])
            cmax = self.cos_near if near else self.cos_cmd
            smax = self.sin_near if near else self.sin_cmd
            if z_des[2] < cmax:                    # clamp commanded tilt
                hz = z_des[:2] / max(np.linalg.norm(z_des[:2]), 1e-12)
                z_des = np.array([hz[0] * smax, hz[1] * smax, cmax])
        else:
            z_des = np.array([0.0, 0.0, 1.0])

        # ---- attitude ------------------------------------------------------------------
        zp = self.thrust_axis(est)
        R = _quat_to_mat(np.asarray(est["torso_quat"], float))     # world <- torso
        w_b = np.asarray(est["torso_angvel"], float)
        ax = np.cross(zp, z_des)                                   # world rotation axis
        s = np.linalg.norm(ax)
        ang = math.atan2(s, float(zp @ z_des))
        e_w = ax / s * ang if s > 1e-9 else np.zeros(3)
        e_b = R.T @ e_w                                            # body-frame error
        tau_b = np.zeros(3)
        for k in range(2):
            ek = e_b[k]
            w_lin = self.kp[k] / self.kd[k] * ek
            w_sq = math.copysign(math.sqrt(2.0 * self.alpha[k] * abs(ek)), ek)
            w_des = w_lin if abs(w_lin) < abs(w_sq) else w_sq
            tau_b[k] = self.kd[k] * (w_des - w_b[k])
        tau_b[2] = -float(p["yaw_rate_kd"]) * w_b[2]
        if p["yaw_align"]:
            # yaw so that the roll/pitch demand lies on the strong pitch axis (jets give
            # ~+-200 N*m pitch but only ~+-50 N*m roll without pod thrust)
            dxy = np.array([self.kp[0] / self.kd[0] * e_b[0] - w_b[0],
                            self.kp[1] / self.kd[1] * e_b[1] - w_b[1]])
            if ang > math.radians(float(p["yaw_align_min_deg"])):
                beta = math.atan2(dxy[1], dxy[0])           # demand direction in body xy
                delta = (beta - math.pi / 2 + math.pi / 2) % math.pi - math.pi / 2  # to +-y
                k = float(p["yaw_align_kp"])
                w_des = math.copysign(min(k * abs(delta), math.sqrt(
                    2.0 * float(p["yaw_align_alpha"]) * abs(delta))), delta)
                tau_b[2] = float(p["yaw_align_kd"]) * (w_des - w_b[2])
        tau_b = np.clip(tau_b, -self.tmax, self.tmax)
        tau_w = R @ tau_b

        # ---- thrust magnitude ----------------------------------------------------------
        # weighted LS along the actual thrust axis zp:
        #   min_T  w_z (T zp_z - Fz_des)^2 + w_h |T zp_xy - F_xy_des|^2
        # (w_h = 0 -> full vertical compensation T = Fz/zp_z; w_h = w_z -> projection)
        th_p = math.acos(max(-1.0, min(1.0, float(zp[2]))))
        wh = float(p["thrust_horizontal_weight"])
        if Fz_des > 0 and zp[2] > 1e-3 and (on_profile or self.braking):
            num = zp[2] * Fz_des + wh * float(zp[:2] @ F_des[:2])
            den = zp[2] * zp[2] + wh * float(zp[:2] @ zp[:2])
            T = max(num / den, 0.0)
            if th_p > self.tilt_full:
                T *= max(0.0, (self.tilt_cut - th_p) / (self.tilt_cut - self.tilt_full))
        else:
            T = 0.0                                   # free fall above the profile
        # roll-authority assist: the pods' differential needs collective thrust; limit the
        # horizontal acceleration it causes unless it points against the horizontal velocity
        excess = abs(tau_b[0]) - float(p["roll_jet_authority"])
        T_assist = min(max(excess, 0.0) / float(p["pod_arm"]), float(p["assist_thrust_max"]))
        sxy = float(np.linalg.norm(zp[:2]))
        if T_assist > 0 and sxy > 1e-6:
            # component of the thrust's horizontal accel along the desired horizontal accel
            helpful = float(zp[:2] @ a_xy) > 0.0
            if not helpful:
                T_assist = min(T_assist, m * float(p["assist_horizontal_accel_max"]) / sxy)
        T = min(max(T, T_assist), float(p["thrust_max"]))
        # keep the torque request feasible for this collective thrust: an infeasible request
        # makes the least-squares allocator trade force rows for torque (extra thrust ->
        # horizontal force) while still under-delivering the torque
        if p["torque_feasible_clip"]:
            rmax = float(p["roll_jet_authority"]) + float(p["pod_arm"]) * T
            tau_b[0] = float(np.clip(tau_b[0], -rmax, rmax))
            tau_w = R @ tau_b
        phase = "brake" if self.braking else "attitude"
        F = T * zp
        # horizontal force the tilted thrust does not deliver -> requested from the jets
        # (fore/aft jets give ~+-400 N along body x without lift)
        kj = float(p["jet_horizontal_gain"])
        if kj > 0:
            F[:2] += kj * (F_des[:2] - F[:2])
            fh = float(np.linalg.norm(F[:2] - T * zp[:2]))
            fmax = float(p["jet_horizontal_force_max"])
            if fh > fmax:
                F[:2] = T * zp[:2] + (F[:2] - T * zp[:2]) * (fmax / fh)
        wrench = np.concatenate([F, tau_w])
        return ControlOutput(wrench=wrench, deploy_airbags=self.airbags, phase=phase,
                             debug={"v_ref": vr, "T": T, "T_assist": T_assist,
                                    "tilt_err": ang, "integ": self.integ,
                                    "z_des": z_des.tolist(), "z_thrust": zp.tolist(),
                                    "tau_b": tau_b.tolist()})
