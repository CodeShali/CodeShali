"""Suit sensor suite: IMUs, radar altimeter, foot pressure, joint encoders, thruster telemetry.

``record()`` runs every physics step and buffers TRUE values (per-sensor delay lines with
the delay rounded to whole physics steps).  ``read()`` returns the delayed samples with
Gaussian noise drawn at read time from the suite's generator.

Altimeter ray filtering
-----------------------
The radar must see the environment (ground, platforms, beams) but never the wearer.  At
construction the suite classifies geoms as *human* (belonging to the kinematic tree of
the wearer, i.e. sharing ``body_rootid`` with the root body) or *environment*.  If the
two sets use disjoint geom groups (A1 convention: environment in group 0, human >= 1) a
single ``mj_ray`` with a geom-group mask is cast.  Otherwise the ray is re-cast past every
human geom it hits (bounded number of passes), so the result never depends on the
grouping convention.
"""
from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from suitsim.interfaces import (
    ALTIMETER_SITE, BUDGET_KEYS, FEET, ROOT_BODY, SEGMENTS, foot_touch_sensor,
    imu_accel_sensor, imu_gyro_sensor, load_thruster_specs,
)
from suitsim.sensors.noise import DelayLine, GaussianNoise, delay_steps

_MAX_RAY_PASSES = 8


def _sensor_adr(model, name: str, dim: int) -> np.ndarray:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        raise KeyError(f"sensor '{name}' not found in model")
    if model.sensor_dim[sid] != dim:
        raise ValueError(f"sensor '{name}' has dim {model.sensor_dim[sid]}, expected {dim}")
    adr = int(model.sensor_adr[sid])
    return np.arange(adr, adr + dim)


def hinge_joint_indices(model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(joint ids, qpos adr, dof adr) of all hinge joints, model order."""
    jids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    return jids, model.jnt_qposadr[jids].astype(int), model.jnt_dofadr[jids].astype(int)


def wearer_root_id(model) -> int:
    """Root-body id of the wearer's kinematic tree."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY)
    if bid < 0:   # stand-in models: use the altimeter's body
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ALTIMETER_SITE)
        bid = int(model.site_bodyid[sid]) if sid >= 0 else 1
    return int(model.body_rootid[bid])


class Altimeter:
    """Radar altimeter ray cast that ignores the wearer's own geoms."""

    def __init__(self, model, cfg_alt: dict):
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ALTIMETER_SITE)
        if self.site_id < 0:
            raise KeyError(f"site '{ALTIMETER_SITE}' not found in model")
        self.body_id = int(model.site_bodyid[self.site_id])
        d = np.asarray(cfg_alt.get("dir", [0, 0, -1]), dtype=float)
        self.dir_body = d / np.linalg.norm(d)
        self.max_range = float(cfg_alt.get("max_range", 30.0))
        root = wearer_root_id(model)
        human_geom = model.body_rootid[model.geom_bodyid] == root
        self.human_body = model.body_rootid == root
        hg = set(model.geom_group[human_geom].tolist())
        eg = set(model.geom_group[~human_geom].tolist())
        if eg and not (hg & eg):
            self.geomgroup = np.array([1 if g in eg else 0 for g in range(6)], dtype=np.uint8)
            self.single_pass = True
        else:
            self.geomgroup = None
            self.single_pass = False
        self._gid = np.zeros(1, dtype=np.int32)
        self.last_geom = -1

    def world_dir(self, data) -> np.ndarray:
        return data.xmat[self.body_id].reshape(3, 3) @ self.dir_body

    def cast(self, model, data) -> float:
        """True slant range [m] along the ray (``max_range`` if nothing within range)."""
        pnt = data.site_xpos[self.site_id].copy()
        vec = self.world_dir(data)
        gid = self._gid
        if self.single_pass:
            dist = mujoco.mj_ray(model, data, pnt, vec, self.geomgroup, 1, self.body_id, gid)
            self.last_geom = int(gid[0])
            return self.max_range if dist < 0 or dist > self.max_range else float(dist)
        total, exclude = 0.0, self.body_id
        for _ in range(_MAX_RAY_PASSES):
            dist = mujoco.mj_ray(model, data, pnt, vec, None, 1, exclude, gid)
            if dist < 0:
                break
            g = int(gid[0])
            b = int(model.geom_bodyid[g])
            if not self.human_body[b]:
                total += dist
                self.last_geom = g
                return self.max_range if total > self.max_range else float(total)
            step = dist + 1e-6
            total += step
            pnt = pnt + step * vec
            exclude = b
        self.last_geom = -1
        return self.max_range


class SensorSuite:
    """Implements :class:`suitsim.interfaces.SensorSuiteProtocol`."""

    def __init__(self, model, cfg: dict, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng(cfg.get("seed", 0))
        sc = cfg["sensors"]
        dt = float(model.opt.timestep)
        self.dt = dt

        # IMUs ---------------------------------------------------------------------------
        imu = sc["imu"]
        segs = imu.get("segments", "all")
        self.segments = tuple(SEGMENTS) if segs in ("all", None) else tuple(segs)
        self._acc_idx = np.stack([_sensor_adr(model, imu_accel_sensor(s), 3)
                                  for s in self.segments])
        self._gyr_idx = np.stack([_sensor_adr(model, imu_gyro_sensor(s), 3)
                                  for s in self.segments])
        nseg = len(self.segments)
        self._imu_steps = delay_steps(imu.get("delay", 0.0), dt)
        self._acc_noise_std = float(imu.get("accel_noise", 0.0))
        self._gyr_noise_std = float(imu.get("gyro_noise", 0.0))

        # Altimeter -------------------------------------------------------------------------
        alt = sc["altimeter"]
        self.altimeter = Altimeter(model, alt)
        self._alt_steps = delay_steps(alt.get("delay", 0.0), dt)
        self._alt_frac = float(alt.get("noise_frac", 0.0))

        # Foot pressure ---------------------------------------------------------------------
        fp = sc["foot_pressure"]
        self._touch_idx = np.concatenate([_sensor_adr(model, foot_touch_sensor(f), 1)
                                          for f in FEET])
        self._fp_steps = delay_steps(fp.get("delay", 0.0), dt)
        self._fp_frac = float(fp.get("noise_frac", 0.0))

        # Joint encoders -------------------------------------------------------------------
        je = sc["joint_encoders"]
        _, self._qadr, self._dadr = hinge_joint_indices(model)
        nj = len(self._qadr)
        self._je_steps = delay_steps(je.get("delay", 0.0), dt)
        self._je_pos_std = float(je.get("noise", 0.0))
        # velocity noise is not in the default YAML; key `sensors.joint_encoders.vel_noise`
        self._je_vel_std = float(je.get("vel_noise", 10.0 * self._je_pos_std))

        # Thruster telemetry --------------------------------------------------------------
        tt = sc.get("thruster_telemetry", {}) or {}
        try:
            self._n_thr = len(load_thruster_specs(cfg))
        except (KeyError, TypeError):
            self._n_thr = 0
        self._thr_steps = delay_steps(tt.get("delay", 0.0), dt)
        self._thr_std = float(tt.get("noise", 0.0))
        b = (cfg.get("suit", {}) or {}).get("budgets", {}) or {}
        self._default_budgets = {k: float(b.get(k, 0.0)) for k in BUDGET_KEYS}

        # delay lines (true values)
        self._acc = DelayLine(self._imu_steps, (nseg, 3))
        self._gyr = DelayLine(self._imu_steps, (nseg, 3))
        self._alt = DelayLine(self._alt_steps, ())
        self._fp = DelayLine(self._fp_steps, (len(FEET),))
        self._jq = DelayLine(self._je_steps, (nj,))
        self._jv = DelayLine(self._je_steps, (nj,))
        self._thr = DelayLine(self._thr_steps, (self._n_thr,))
        self._budget_hist: list[dict] = []
        self._t = 0.0
        self.noise_scale = 1.0
        self._make_noise()

    # ------------------------------------------------------------------------------------
    def _make_noise(self) -> None:
        s = self.noise_scale
        self._n_acc = GaussianNoise(self._acc_noise_std, scale=s)
        self._n_gyr = GaussianNoise(self._gyr_noise_std, scale=s)
        self._n_alt = GaussianNoise(self._alt_frac, relative=True, scale=s)
        self._n_fp = GaussianNoise(self._fp_frac, relative=True, scale=s)
        self._n_jq = GaussianNoise(self._je_pos_std, scale=s)
        self._n_jv = GaussianNoise(self._je_vel_std, scale=s)
        self._nz_thr = GaussianNoise(self._thr_std, scale=s)

    @property
    def n_joints(self) -> int:
        return len(self._qadr)

    def _sample(self, model, data, thrusters, fill: bool) -> None:
        sd = data.sensordata
        t = float(data.time)
        if thrusters is not None:
            forces = np.asarray(thrusters.forces, dtype=float)
            budgets = dict(thrusters.remaining())
        else:
            forces = np.zeros(self._n_thr)
            budgets = dict(self._default_budgets)
        vals = (
            (self._acc, sd[self._acc_idx]), (self._gyr, sd[self._gyr_idx]),
            (self._alt, self.altimeter.cast(model, data)), (self._fp, sd[self._touch_idx]),
            (self._jq, data.qpos[self._qadr]), (self._jv, data.qvel[self._dadr]),
            (self._thr, forces),
        )
        for line, v in vals:
            (line.fill if fill else line.push)(v, t)
        if fill:
            self._budget_hist = [budgets] * (self._thr_steps + 1)
        else:
            self._budget_hist.append(budgets)
            if len(self._budget_hist) > self._thr_steps + 1:
                del self._budget_hist[0]
        self._t = t

    def reset(self, model, data, noise_scale: float = 1.0) -> None:
        """Clear history and pre-fill every delay line with the current true values."""
        self.noise_scale = float(noise_scale)
        self._make_noise()
        mujoco.mj_forward(model, data)   # derived quantities only; state unchanged
        self._sample(model, data, None, fill=True)

    def record(self, model, data, thrusters: Any) -> None:
        self._sample(model, data, thrusters, fill=False)

    def read(self) -> dict:
        rng = self.rng
        alt = float(self._n_alt.apply(self._alt.get(), rng))
        mr = self.altimeter.max_range
        alt_true = float(self._alt.get())
        alt = mr if alt_true >= mr else float(np.clip(alt, 0.0, mr))
        return {
            "t": self._t,
            "imu_accel": self._n_acc.apply(self._acc.get(), rng),
            "imu_gyro": self._n_gyr.apply(self._gyr.get(), rng),
            "altimeter": alt,
            "foot_pressure": np.maximum(self._n_fp.apply(self._fp.get(), rng), 0.0),
            "joint_pos": self._n_jq.apply(self._jq.get(), rng),
            "joint_vel": self._n_jv.apply(self._jv.get(), rng),
            "thruster_force": np.maximum(self._nz_thr.apply(self._thr.get(), rng), 0.0),
            "budgets": dict(self._budget_hist[0]),
        }
