"""State estimator (workstream A3) — sees only sensor readings.

Algorithm (per ``update``, 500 Hz)
---------------------------------
1. **Attitude**: torso gyro (rotated from the IMU site frame into the torso body frame)
   integrated with trapezoidal rates on the unit quaternion.  When the body is
   quasi-static (|mass-weighted specific force| ~ g and low gyro) a complementary
   correction pulls the estimated gravity direction onto the measured one (tilt only,
   yaw is unobservable).
2. **Kinematics**: own ``MjData`` posed from the estimated torso attitude + measured joint
   angles -> CoM, IMU frames, altimeter pose, soles, thruster sites.
3. **CoM acceleration**: mass-weighted average of all 15 IMU specific forces (centripetal
   lever-arm term removed using each gyro), rotated to world with each IMU's estimated
   orientation, plus gravity.  In free fall without thrust this is ~ -g ẑ exactly
   (internal forces cancel), independent of attitude error.
4. **Vertical channel**: 2-state Kalman filter [CoM height, vertical velocity] driven by
   the CoM acceleration; measurement = altimeter slant range x cos(ray, vertical) converted
   to CoM height through kinematics, with the altimeter/IMU delay difference compensated.
   Measurement noise from the configured 2 % range noise (+ attitude-induced error);
   innovation gating with re-acquisition (a new surface below, e.g. a platform, is
   adopted after it persists).
5. **Horizontal velocity**: integrated CoM acceleration; zero-velocity update when
   quasi-static with foot contact.
6. The state is propagated in real time from the reset instant, so outputs refer to the
   reading time; the older altimeter sample is advanced by ``v_z * delay``.
7. **Free fall**: |torso specific force| < ``freefall_accel_thresh`` continuously for
   ``freefall_min_time`` -> latched ``free_fall``; ``time_since_fall`` is the time since
   the start of that low-g window (0 before detection).
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from suitsim.interfaces import BUDGET_KEYS, FEET, SEGMENTS, TORSO_BODY
from suitsim.estimation.kinematics import KinematicsModel, KinState, quat_mul, quat_to_mat

# Defaults for keys not (yet) present in configs/suit_default.yaml -> estimation.*
DEFAULTS = {
    "foot_contact_threshold": 20.0,      # [N]
    "height_filter": "kf",               # 'kf' | 'complementary' (uses altimeter_gain)
    "accel_process_noise": 0.5,          # [m/s^2] std of CoM-acceleration model error
    "attitude_gain": 2.0,                # [1/s] quasi-static tilt correction gain
    "zupt_gain": 2.0,                    # [1/s] horizontal velocity decay when static
    "static_accel_tol": 0.4,             # [m/s^2] | |f| - g | below => quasi-static
    "static_gyro_tol": 0.3,              # [rad/s]
    "altimeter_max_incidence_deg": 60.0,  # reject rays more oblique than this
    "altimeter_gate_sigma": 5.0,         # innovation gate [sigma]
    "altimeter_reacquire_time": 0.1,     # [s] of consecutive rejects -> adopt new surface
    "kinematic_height_noise": 0.01,      # [m] std, CoM-from-site kinematic error
    "attitude_err_corr": 100.0,          # [samples] correlation of attitude-induced alt error
}


def _random_unit(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(3)
    return v / max(np.linalg.norm(v), 1e-12)


class StateEstimator:
    """Implements :class:`suitsim.interfaces.EstimatorProtocol`."""

    def __init__(self, model, cfg: dict):
        self.cfg = cfg
        ec = dict(DEFAULTS)
        ec.update(cfg.get("estimation", {}) or {})
        self.ec = ec
        sc = cfg.get("sensors", {}) or {}
        self.g = float(cfg.get("sim", {}).get("gravity", 9.81))
        self.kin = KinematicsModel(model, cfg)
        self.n_joints = self.kin.n_joints
        self.d_imu = float(sc.get("imu", {}).get("delay", 0.0))
        self.d_alt = float(sc.get("altimeter", {}).get("delay", 0.0))
        alt = sc.get("altimeter", {}) or {}
        d = np.asarray(alt.get("dir", [0, 0, -1]), dtype=float)
        self.alt_dir_body = d / np.linalg.norm(d)
        self.alt_max = float(alt.get("max_range", 30.0))
        self.alt_frac = float(alt.get("noise_frac", 0.02))
        self.cos_min = math.cos(math.radians(float(ec["altimeter_max_incidence_deg"])))
        self.att_sigma = math.radians(max(float(ec.get("init_attitude_err_deg", 2.0)), 0.5))

        w = self.kin.imu_mass
        self.w_imu = w / w.sum()
        self.lever = self.kin.imu_lever                      # (15,3) site frame
        self.R_ts = self.kin.R_torso_site
        self.ti = self.kin.torso_imu
        self._budgets0 = {k: float((cfg.get("suit", {}) or {}).get("budgets", {}).get(k, 0.0))
                          for k in BUDGET_KEYS}
        self.reset({"torso_quat": np.array([1.0, 0, 0, 0]), "com_vel": np.zeros(3),
                    "com_height": 1.0}, np.random.default_rng(0), apply_errors=False)

    # ------------------------------------------------------------------------------------
    def reset(self, init_hint: dict, rng: np.random.Generator, apply_errors: bool = True) -> None:
        ec = self.ec
        q = np.asarray(init_hint["torso_quat"], dtype=float).copy()
        v = np.asarray(init_hint.get("com_vel", np.zeros(3)), dtype=float).copy()
        h = float(init_hint.get("com_height", 1.0))
        e_att = math.radians(float(ec.get("init_attitude_err_deg", 0.0)))
        e_v = float(ec.get("init_velocity_err", 0.0))
        e_h = float(ec.get("init_height_err", 0.0))
        if apply_errors:
            # fixed-magnitude errors in random directions (bounded, reproducible)
            dq = np.empty(4)
            mujoco.mju_axisAngle2Quat(dq, _random_unit(rng), e_att)
            q = quat_mul(dq, q)
            v = v + e_v * _random_unit(rng)
            h = h + e_h * (1.0 if rng.random() < 0.5 else -1.0)
        self.q = q / np.linalg.norm(q)
        self.v = v
        self.x = np.array([h, v[2]])
        self.P = np.diag([max(e_h, 0.01) ** 2, max(e_v, 0.01) ** 2])
        self.prev_w = None
        self.t_prev = None
        self.low_g_start = None
        self.free_fall = False
        self.t_fall_onset = 0.0
        self.n_reject = 0
        self.a_com = np.array([0.0, 0.0, 0.0])
        self.last_kin: KinState | None = None
        self.last_est: dict | None = None
        self.quasi_static = False

    # ------------------------------------------------------------------------------------
    def update(self, sensors: dict, dt: float) -> dict:
        ec, g = self.ec, self.g
        t = float(sensors["t"])
        acc = np.asarray(sensors["imu_accel"], dtype=float)
        gyr = np.asarray(sensors["imu_gyro"], dtype=float)
        jq = np.asarray(sensors["joint_pos"], dtype=float)

        # 1. attitude propagation (trapezoidal body rate)
        w_t = self.R_ts @ gyr[self.ti]
        w_mid = w_t if self.prev_w is None else 0.5 * (w_t + self.prev_w)
        self.prev_w = w_t
        mujoco.mju_quatIntegrate(self.q, w_mid, dt)
        mujoco.mju_normalize4(self.q)

        # 2. kinematics at the current attitude estimate
        kin = self.kin.compute(self.q, jq)

        # 3. CoM acceleration from all IMUs (lever-arm centripetal term removed)
        # w x (w x r) = w (w.r) - r |w|^2   (np.cross is slow for tiny arrays)
        lev = self.lever
        cen = gyr * np.einsum("ki,ki->k", gyr, lev)[:, None] \
            - lev * np.einsum("ki,ki->k", gyr, gyr)[:, None]
        f_site = acc + cen          # site acc = segCoM acc - w x (w x r_site->com) (no alpha)
        f_w = np.einsum("k,kij,kj->i", self.w_imu, kin.imu_R, f_site)
        a = f_w.copy()
        a[2] -= g
        self.a_com = a
        fmag = math.sqrt(f_w @ f_w)
        wmag = math.sqrt(w_t @ w_t)
        self.quasi_static = (abs(fmag - g) < ec["static_accel_tol"]
                             and wmag < ec["static_gyro_tol"])

        fp = np.asarray(sensors["foot_pressure"], dtype=float)
        contact = fp > float(ec["foot_contact_threshold"])

        # 1b. quasi-static tilt correction (world-frame small rotation)
        if self.quasi_static and fmag > 1e-6:
            fx, fy = f_w[0] / fmag, f_w[1] / fmag
            axis = np.array([fy, -fx, 0.0])               # f_hat x z_hat
            ang = float(ec["attitude_gain"]) * dt
            dq = np.empty(4)
            n = np.linalg.norm(axis)
            if n > 1e-9:
                mujoco.mju_axisAngle2Quat(dq, axis / n, ang * math.asin(min(n, 1.0)))
                self.q = quat_mul(dq, self.q)
                mujoco.mju_normalize4(self.q)

        # 4. vertical channel
        self._vertical(a[2], sensors, kin, dt)

        # 5. horizontal velocity
        self.v[:2] += a[:2] * dt
        if self.quasi_static and contact.any():
            self.v[:2] *= max(0.0, 1.0 - float(ec["zupt_gain"]) * dt)
        self.v[2] = self.x[1]

        # 7. free fall detection (torso specific force)
        ft = acc[self.ti]
        if math.sqrt(ft @ ft) < float(ec["freefall_accel_thresh"]):
            if self.low_g_start is None:
                self.low_g_start = t
            if not self.free_fall and t - self.low_g_start >= float(ec["freefall_min_time"]) - 1e-9:
                self.free_fall = True
                self.t_fall_onset = self.low_g_start
        else:
            self.low_g_start = None

        # 6. outputs (the state is integrated in real time from the reset instant, so it
        #    already refers to the reading time; IMU delay only lags the driving inputs)
        q_out = self.q.copy()
        h_out = self.x[0]
        v_out = self.v.copy()
        com_z = kin.com[2]
        feet_h = h_out - (com_z - kin.sole_pos[:, 2].min())
        Rt = quat_to_mat(q_out)
        tilt = math.acos(max(-1.0, min(1.0, Rt[2, 2])))
        budgets = sensors.get("budgets")
        est = {
            "t": t,
            "com_height": float(h_out),
            "feet_height": float(feet_h),
            "com_vel": v_out,
            "torso_quat": q_out,
            "torso_angvel": w_t,
            "tilt": tilt,
            "joint_pos": jq.copy(),
            "joint_vel": np.asarray(sensors["joint_vel"], dtype=float).copy(),
            "budgets": dict(budgets) if budgets is not None else dict(self._budgets0),
            "free_fall": bool(self.free_fall),
            "time_since_fall": float(t - self.t_fall_onset) if self.free_fall else 0.0,
            "foot_contact": contact,
        }
        self.last_kin = kin
        self.last_est = est
        self.t_prev = t
        return est

    # ------------------------------------------------------------------------------------
    def _vertical(self, az: float, sensors: dict, kin: KinState, dt: float) -> None:
        ec = self.ec
        x, P = self.x, self.P
        # predict
        x[0] += x[1] * dt + 0.5 * az * dt * dt
        x[1] += az * dt
        qa = float(ec["accel_process_noise"]) ** 2
        dt2 = dt * dt
        F01 = dt
        P00 = P[0, 0] + 2 * F01 * P[0, 1] + dt2 * P[1, 1] + qa * dt2 * dt2 / 4
        P01 = P[0, 1] + F01 * P[1, 1] + qa * dt2 * dt / 2
        P11 = P[1, 1] + qa * dt2
        P[0, 0], P[0, 1], P[1, 0], P[1, 1] = P00, P01, P01, P11

        r = float(sensors["altimeter"])
        if not (0.0 < r < self.alt_max * 0.999):
            return
        ray = kin.alt_R @ self.alt_dir_body
        c = -ray[2]
        if c < self.cos_min:
            return
        h_site = r * c
        z = h_site + (kin.com[2] - kin.alt_pos[2])
        z += x[1] * self.d_alt                         # measurement is d_alt older than state
        s_in = math.sqrt(max(0.0, 1.0 - c * c))
        # attitude error biases the slant->vertical conversion coherently over many
        # samples; inflate its variance by the number of samples it stays correlated
        Rm = ((self.alt_frac * r * c) ** 2
              + float(ec["attitude_err_corr"]) * (r * s_in * self.att_sigma) ** 2
              + float(ec["kinematic_height_noise"]) ** 2)
        if ec.get("height_filter", "kf") == "complementary":
            k = float(ec.get("altimeter_gain", 0.2))
            innov = z - x[0]
            x[0] += k * innov
            x[1] += (k * k / (2.0 - k)) * innov / max(dt, 1e-6)   # alpha-beta filter
            return
        S = P[0, 0] + Rm
        innov = z - x[0]
        if innov * innov > (float(ec["altimeter_gate_sigma"]) ** 2) * S:
            self.n_reject += 1
            if self.n_reject * dt >= float(ec["altimeter_reacquire_time"]):
                x[0] = z                                  # new surface below: adopt it
                P[0, 0] = Rm
                P[0, 1] = P[1, 0] = 0.0
                self.n_reject = 0
            return
        self.n_reject = 0
        K0 = P[0, 0] / S
        K1 = P[1, 0] / S
        x[0] += K0 * innov
        x[1] += K1 * innov
        P00, P01, P11 = P[0, 0], P[0, 1], P[1, 1]
        P[0, 0] = (1 - K0) * P00
        P[0, 1] = P[1, 0] = (1 - K0) * P01
        P[1, 1] = P11 - K1 * P01

    # ------------------------------------------------------------------------------------
    def thruster_geometry(self, est: dict) -> tuple[np.ndarray, np.ndarray]:
        if self.kin.thr_site is None:
            raise RuntimeError("model has no thruster sites matching the config specs")
        kin = self.last_kin
        if kin is None or est is not self.last_est:
            kin = self.kin.compute(np.asarray(est["torso_quat"], dtype=float),
                                   np.asarray(est["joint_pos"], dtype=float))
        return kin.thr_pos - kin.com, kin.thr_dir.copy()
