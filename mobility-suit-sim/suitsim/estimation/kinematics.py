"""Forward kinematics of the wearer from *estimated* torso attitude + *measured* joints.

Uses its own ``MjData`` (never the simulation's).  All returned positions are
world-ALIGNED vectors relative to the root-body origin (the absolute position is unknown
to the estimator and irrelevant: everything downstream uses differences / heights
relative to the altimeter).
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from suitsim.interfaces import (
    ALTIMETER_SITE, FEET, ROOT_BODY, SEGMENTS, TORSO_BODY, foot_sole_site,
    imu_accel_sensor, load_thruster_specs,
)


def _id(model, objtype, name: str, required: bool = True) -> int:
    i = mujoco.mj_name2id(model, objtype, name)
    if i < 0 and required:
        raise KeyError(f"{objtype.name} '{name}' not found in model")
    return int(i)


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    res = np.empty(4)
    mujoco.mju_mulQuat(res, a, b)
    return res


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    m = np.empty(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


@dataclass
class KinState:
    """Kinematic snapshot (world-aligned, origin = root-body origin)."""
    R_root: np.ndarray        # (3,3) pelvis orientation
    com: np.ndarray           # (3,) whole-body CoM
    imu_R: np.ndarray         # (15,3,3) IMU (accelerometer site) frames, world<-site
    alt_pos: np.ndarray       # (3,) altimeter site position
    alt_R: np.ndarray         # (3,3) altimeter BODY orientation (ray dir is in body frame)
    sole_pos: np.ndarray      # (2,3) sole sites, order FEET
    thr_pos: np.ndarray | None   # (N,3) thruster sites
    thr_dir: np.ndarray | None   # (N,3) thruster site z axes


class KinematicsModel:
    def __init__(self, model, cfg: dict | None = None):
        self.model = model
        self.data = mujoco.MjData(model)
        m = model
        OBJ = mujoco.mjtObj
        self.root_id = _id(m, OBJ.mjOBJ_BODY, ROOT_BODY)
        self.torso_id = _id(m, OBJ.mjOBJ_BODY, TORSO_BODY)
        free = [j for j in range(m.njnt)
                if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and m.jnt_bodyid[j] == self.root_id]
        if not free:
            raise ValueError(f"root body '{ROOT_BODY}' has no free joint")
        self.root_qadr = int(m.jnt_qposadr[free[0]])
        hinge = np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
        self.hinge_qadr = m.jnt_qposadr[hinge].astype(int)
        self.n_joints = len(hinge)

        # wearer bodies + masses ----------------------------------------------------------
        rootid = int(m.body_rootid[self.root_id])
        self.human_bodies = np.flatnonzero(m.body_rootid == rootid)
        self.body_mass = m.body_mass[self.human_bodies].copy()
        self.total_mass = float(self.body_mass.sum())

        # IMU sites (from the accelerometer sensors, so the frame is the sensor frame) -----
        self.imu_site = np.array([
            int(m.sensor_objid[_id(m, OBJ.mjOBJ_SENSOR, imu_accel_sensor(s))]) for s in SEGMENTS])
        self.imu_body = m.site_bodyid[self.imu_site].astype(int)
        self.torso_imu = SEGMENTS.index(TORSO_BODY)

        # altimeter, soles, thrusters -------------------------------------------------------
        self.alt_site = _id(m, OBJ.mjOBJ_SITE, ALTIMETER_SITE)
        self.alt_body = int(m.site_bodyid[self.alt_site])
        self.sole_site = np.array([_id(m, OBJ.mjOBJ_SITE, foot_sole_site(f)) for f in FEET])
        self.thr_site = None
        if cfg is not None:
            try:
                specs = load_thruster_specs(cfg)
            except (KeyError, TypeError):
                specs = []
            ids = [_id(m, OBJ.mjOBJ_SITE, s.site, required=False) for s in specs]
            if specs and all(i >= 0 for i in ids):
                self.thr_site = np.array(ids)

        self._segment_mass_model()

    # ------------------------------------------------------------------------------------
    def _segment_mass_model(self) -> None:
        """Lump every wearer body's mass onto its nearest IMU-carrying ancestor; return
        per-IMU mass and the lumped CoM expressed in the IMU site frame (lever arm used to
        remove centripetal terms from the accelerometer readings)."""
        m, d = self.model, self.data
        d.qpos[:] = m.qpos0
        mujoco.mj_kinematics(m, d)
        imu_of_body = {}
        for k, b in enumerate(self.imu_body):
            imu_of_body.setdefault(int(b), k)
        n = len(SEGMENTS)
        mass = np.zeros(n)
        mcom = np.zeros((n, 3))
        for b, bm in zip(self.human_bodies, self.body_mass):
            a = int(b)
            while a not in imu_of_body and a != 0:
                a = int(m.body_parentid[a])
            if a not in imu_of_body:
                continue
            k = imu_of_body[a]
            mass[k] += bm
            mcom[k] += bm * d.xipos[b]
        com = mcom / np.maximum(mass, 1e-12)[:, None]
        Rs = d.site_xmat[self.imu_site].reshape(-1, 3, 3)
        # lever arm site->segment CoM in site frame
        self.imu_lever = np.einsum("kji,kj->ki", Rs, com - d.site_xpos[self.imu_site])
        self.imu_mass = mass
        # torso IMU frame relative to torso body (for body-frame angular velocity)
        Rb = d.xmat[self.torso_id].reshape(3, 3)
        self.R_torso_site = Rb.T @ Rs[self.torso_imu]       # torso_body <- site

    # ------------------------------------------------------------------------------------
    def compute(self, torso_quat: np.ndarray, joint_pos: np.ndarray) -> KinState:
        m, d = self.model, self.data
        a = self.root_qadr
        d.qpos[a:a + 3] = 0.0
        d.qpos[a + 3:a + 7] = (1.0, 0.0, 0.0, 0.0)
        d.qpos[self.hinge_qadr] = joint_pos
        mujoco.mj_kinematics(m, d)
        q_rel = d.xquat[self.torso_id]                     # torso orientation w.r.t. pelvis
        q_root = quat_mul(torso_quat, quat_conj(q_rel))
        R = quat_to_mat(q_root)
        com_l = self.body_mass @ d.xipos[self.human_bodies] / self.total_mass
        RT = R.T
        imu_R = R @ d.site_xmat[self.imu_site].reshape(-1, 3, 3)
        thr_pos = thr_dir = None
        if self.thr_site is not None:
            thr_pos = d.site_xpos[self.thr_site] @ RT
            thr_dir = d.site_xmat[self.thr_site].reshape(-1, 3, 3)[:, :, 2] @ RT
        return KinState(
            R_root=R, com=R @ com_l, imu_R=imu_R,
            alt_pos=R @ d.site_xpos[self.alt_site],
            alt_R=R @ d.xmat[self.alt_body].reshape(3, 3),
            sole_pos=d.site_xpos[self.sole_site] @ RT,
            thr_pos=thr_pos, thr_dir=thr_dir,
        )
