"""Initial-state placement and ground-truth whole-body quantities."""
from __future__ import annotations

import mujoco
import numpy as np

from suitsim.interfaces import ROOT_BODY, load_config


def _hinges(model: mujoco.MjModel) -> list[int]:
    h = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    return sorted(h, key=lambda j: model.jnt_qposadr[j])


def nominal_posture(model: mujoco.MjModel, cfg: dict | None = None) -> np.ndarray:
    """Nominal hinge angles [rad] in hinge qpos order.

    From ``cfg['human']['posture']`` if cfg is given, otherwise from the model's ``stand``
    keyframe, otherwise from the default config.
    """
    hinges = _hinges(model)
    if cfg is None:
        from suitsim.model.builder import STAND_KEY
        k = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, STAND_KEY)
        if k >= 0:
            return np.array([model.key_qpos[k, model.jnt_qposadr[j]] for j in hinges])
        cfg = load_config()
    posture = cfg["human"].get("posture", {}) or {}
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in hinges]
    return np.deg2rad([float(posture.get(n, 0.0)) for n in names])


def apply_posture_ctrl(model: mujoco.MjModel, data: mujoco.MjData,
                       posture: np.ndarray | None = None) -> None:
    """Set every posture servo's setpoint (ctrl) to the nominal (or given) posture."""
    from suitsim.model.builder import servo_actuator
    posture = nominal_posture(model) if posture is None else posture
    for j, q in zip(_hinges(model), posture):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, servo_actuator(name))
        if a >= 0:
            data.ctrl[a] = q


def _geom_lowest_z(model: mujoco.MjModel, data: mujoco.MjData, g: int) -> float:
    """Lowest world z of geom g at the current kinematics."""
    t = model.geom_type[g]
    s = model.geom_size[g]
    R = data.geom_xmat[g].reshape(3, 3)
    z = data.geom_xpos[g, 2]
    T = mujoco.mjtGeom
    if t == T.mjGEOM_SPHERE:
        return z - s[0]
    if t == T.mjGEOM_CAPSULE:
        return z - abs(R[2, 2]) * s[1] - s[0]
    if t == T.mjGEOM_CYLINDER:
        return z - abs(R[2, 2]) * s[1] - s[0] * np.hypot(R[2, 0], R[2, 1])
    if t == T.mjGEOM_BOX:
        return z - float(np.abs(R[2, :]) @ s[:3])
    if t == T.mjGEOM_ELLIPSOID:
        return z - float(np.linalg.norm(R[2, :] * s[:3]))
    return z - model.geom_rbound[g]


def lowest_point(model: mujoco.MjModel, data: mujoco.MjData,
                 include_disabled: bool = False) -> float:
    """Lowest world z over all collision-enabled body geoms (world-body geoms excluded)."""
    zmin = np.inf
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0:
            continue
        if not include_disabled and model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        zmin = min(zmin, _geom_lowest_z(model, data, g))
    return float(zmin)


def set_initial_pose(model: mujoco.MjModel, data: mujoco.MjData, height_feet: float,
                     quat: np.ndarray, angvel_world: np.ndarray,
                     linvel: np.ndarray = np.zeros(3), cfg: dict | None = None) -> None:
    """Place the body for a drop.

    The root orientation is ``quat`` (wxyz, world<-pelvis), hinges are at the nominal
    posture (servo setpoints too), and the root is translated so that the LOWEST point of any
    collision geom is ``height_feet`` above the ground (z=0).  Velocities are zero except the
    root's linear (world) and angular (given in world frame) velocity. Calls mj_forward.
    """
    from suitsim.model.builder import FREE_JOINT
    jf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, FREE_JOINT)
    qa, va = model.jnt_qposadr[jf], model.jnt_dofadr[jf]
    q = np.asarray(quat, float)
    q = q / np.linalg.norm(q)

    posture = nominal_posture(model, cfg)
    data.qpos[:] = model.qpos0
    for j, a in zip(_hinges(model), posture):
        data.qpos[model.jnt_qposadr[j]] = a
    data.qpos[qa:qa + 3] = 0.0
    data.qpos[qa + 3:qa + 7] = q
    data.qvel[:] = 0.0
    data.qacc_warmstart[:] = 0.0
    if model.na:
        data.act[:] = 0.0
    apply_posture_ctrl(model, data, posture)

    mujoco.mj_kinematics(model, data)
    data.qpos[qa + 2] = float(height_feet) - lowest_point(model, data)

    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    data.qvel[va:va + 3] = np.asarray(linvel, float)
    data.qvel[va + 3:va + 6] = R.reshape(3, 3).T @ np.asarray(angvel_world, float)  # body frame
    mujoco.mj_forward(model, data)


def com_state(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth whole-body CoM position and velocity (world frame).

    Uses subtree_com / cvel from the LAST forward pass: right after mj_step these describe
    the state at the start of that step (one dt behind qpos/qvel); call mj_forward first if
    you need the exact current state.
    """
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY)
    mujoco.mj_subtreeVel(model, data)
    return data.subtree_com[root].copy(), data.subtree_linvel[root].copy()
