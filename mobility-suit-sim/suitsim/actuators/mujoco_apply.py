"""Turn thruster forces / aerodynamic drag into MuJoCo body wrenches.

``data.xfrc_applied[b]`` is a world-frame ``[F, T]`` acting at body ``b``'s centre of mass
(``data.xipos[b]``). A force ``F`` at point ``p`` therefore contributes torque
``(p - xipos[b]) x F``. These helpers only *compute* wrenches; the env sums the
contributions (thrusters + drag + anything else) and writes ``xfrc_applied`` once.
"""
from __future__ import annotations

import mujoco
import numpy as np


def thruster_ids(model, specs) -> tuple[np.ndarray, np.ndarray]:
    """(site_ids, body_ids) for the given thruster specs (via their MJCF site names)."""
    sids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s.site) for s in specs],
                    dtype=int)
    if np.any(sids < 0):
        missing = [s.site for s, i in zip(specs, sids) if i < 0]
        raise KeyError(f"thruster sites missing from model: {missing}")
    return sids, model.site_bodyid[sids].astype(int)


def apply_thrusters(model, data, site_ids, body_ids, forces, out=None,
                    accumulate: bool = False) -> np.ndarray:
    """Per-body world wrenches (nbody, 6) from thruster force magnitudes.

    Each thruster pushes along its site's world z-axis with magnitude ``forces[i]`` at the
    site position (requires up-to-date kinematics: site_xpos/site_xmat/xipos).
    Wrenches of thrusters sharing a body are summed. ``out`` (nbody,6) is zeroed first unless
    ``accumulate`` is True. ``body_ids=None`` uses ``model.site_bodyid``.
    """
    site_ids = np.asarray(site_ids, dtype=int)
    if body_ids is None:
        body_ids = model.site_bodyid[site_ids]
    body_ids = np.asarray(body_ids, dtype=int)
    if out is None:
        out = np.zeros((model.nbody, 6))
    elif not accumulate:
        out[:] = 0.0
    f = np.asarray(forces, dtype=float)
    zaxis = data.site_xmat[site_ids].reshape(-1, 3, 3)[:, :, 2]
    F = f[:, None] * zaxis
    r = data.site_xpos[site_ids] - data.xipos[body_ids]
    T = np.cross(r, F)
    np.add.at(out, body_ids, np.hstack((F, T)))
    return out


def body_drag_wrench(model, data, cfg: dict, root_body_id: int,
                     com_vel: np.ndarray | None = None) -> np.ndarray:
    """Quadratic aerodynamic drag force (3,) [N], world frame, on the whole body.

    ``F = -0.5 rho Cd A |v| v`` with ``v`` the CoM velocity of the subtree rooted at
    ``root_body_id`` (computed with ``mj_subtreeVel`` unless ``com_vel`` is given).
    The caller decides where to apply it (e.g. at the root/torso CoM, zero torque).
    """
    sim = cfg["sim"]
    if com_vel is None:
        mujoco.mj_subtreeVel(model, data)
        com_vel = data.subtree_linvel[root_body_id]
    v = np.asarray(com_vel, dtype=float)
    return -0.5 * float(sim["air_density"]) * float(sim["drag_cd"]) * float(sim["drag_area"]) \
        * np.linalg.norm(v) * v
