"""Airbag stow / deploy by geometry.

MuJoCo decides part of the collision filtering at compile time: a geom compiled with
``contype = conaffinity = 0`` cannot be made to collide later (runtime toggling produced no
contacts and ``mj_setConst`` fails with "setBound: geom ... collides but was not compiled to
collide").  The builder therefore compiles every airbag geom with the body-geom contact bits
(collides with the ground only, never with the wearer), and this module stows / deploys
them by **geometry**:

* stowed  (fraction 0): size = ``suit.airbags.stowed_scale`` x deployed size, centred at the
  carrier segment's primary geom centre, i.e. deep inside the body -> never touches the
  ground;
* deployed (fraction 1): the configured ``size`` / ``pos``;
* in between: size and position interpolate linearly with the inflation fraction
  (:class:`AirbagState` inflates over ``suit.airbags.deploy_time``).

``geom_size``, ``geom_pos``, ``geom_rbound`` and ``geom_aabb`` are written together
(``mj_setConst`` keeps them), so broad- and narrow-phase agree.  Airbags are massless, so
no inertia changes.  The altimeter ray ignores every wearer geom (see
``suitsim.sensors.suite``), so airbags in either state never shorten it.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from suitsim.interfaces import AIRBAGS, airbag_geom, load_config

DEPLOYED_ALPHA = 0.55   # cosmetic

_T = mujoco.mjtGeom


def airbag_geom_ids(model: mujoco.MjModel) -> list[int]:
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, airbag_geom(b)) for b in AIRBAGS]
    return [i for i in ids if i >= 0]


def _bounds(gtype: int, size: np.ndarray) -> tuple[float, np.ndarray]:
    """(rbound, aabb half-extents) of a primitive geom in its local frame (MuJoCo rules)."""
    s = np.asarray(size, float)
    if gtype == _T.mjGEOM_SPHERE:
        return float(s[0]), np.array([s[0], s[0], s[0]])
    if gtype == _T.mjGEOM_CAPSULE:
        return float(s[0] + s[1]), np.array([s[0], s[0], s[0] + s[1]])
    if gtype == _T.mjGEOM_CYLINDER:
        return float(math.hypot(s[0], s[1])), np.array([s[0], s[0], s[1]])
    if gtype == _T.mjGEOM_ELLIPSOID:
        return float(s.max()), s.copy()
    if gtype == _T.mjGEOM_BOX:
        return float(np.linalg.norm(s)), s.copy()
    raise ValueError(f"unsupported airbag geom type {gtype}")


def _airbag_cfg(cfg: dict | None) -> dict:
    if cfg is None:
        cfg = _default_cfg()
    return cfg["suit"]["airbags"]


class _Geometry:
    """Deployed / stowed geometry of each airbag geom of one model (from the config)."""

    def __init__(self, model: mujoco.MjModel, cfg: dict | None):
        ac = _airbag_cfg(cfg)
        self.ids, dep_size, dep_pos, stow_pos = [], [], [], []
        for bag in AIRBAGS:
            g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, airbag_geom(bag))
            if g < 0 or bag not in ac:
                continue
            size = np.zeros(3)
            s = np.asarray(ac[bag]["size"], float)
            size[: s.size] = s
            self.ids.append(g)
            dep_size.append(size)
            dep_pos.append(np.asarray(ac[bag]["pos"], float))
            stow_pos.append(_carrier_centre(model, int(model.geom_bodyid[g])))
        self.ids = np.asarray(self.ids, int)
        self.dep_size = np.asarray(dep_size, float).reshape(-1, 3)
        self.dep_pos = np.asarray(dep_pos, float).reshape(-1, 3)
        self.stow_pos = np.asarray(stow_pos, float).reshape(-1, 3)
        self.stowed_scale = float(ac["stowed_scale"])
        rb, half = [], []
        for g, s in zip(self.ids, self.dep_size):
            r, h = _bounds(int(model.geom_type[g]), s)
            rb.append(r)
            half.append(h)
        self.dep_rbound = np.asarray(rb, float)
        self.dep_half = np.asarray(half, float).reshape(-1, 3)

    def apply(self, model: mujoco.MjModel, fraction: float) -> None:
        f = min(1.0, max(0.0, float(fraction)))
        k = self.stowed_scale + f * (1.0 - self.stowed_scale)   # linear size scale
        ids = self.ids
        model.geom_size[ids] = k * self.dep_size
        model.geom_pos[ids] = self.stow_pos + f * (self.dep_pos - self.stow_pos)
        model.geom_rbound[ids] = k * self.dep_rbound
        model.geom_aabb[ids, :3] = 0.0
        model.geom_aabb[ids, 3:] = k * self.dep_half
        model.geom_rgba[ids, 3] = DEPLOYED_ALPHA if f > 0.0 else 0.0

    def fraction(self, model: mujoco.MjModel) -> float:
        if not self.ids.size:
            return 0.0
        k = float(model.geom_size[self.ids[0], 0] / self.dep_size[0, 0])
        return min(1.0, max(0.0, (k - self.stowed_scale) / (1.0 - self.stowed_scale)))


def _carrier_centre(model: mujoco.MjModel, body: int) -> np.ndarray:
    """Centre of the carrier segment's primary (massive, colliding) geom, body frame."""
    from suitsim.model.builder import segment_geom
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, segment_geom(name))
    return model.geom_pos[g].copy() if g >= 0 else np.zeros(3)


_GEOM_CACHE: dict[int, tuple[mujoco.MjModel, dict, _Geometry]] = {}


def _geometry(model: mujoco.MjModel, cfg: dict | None) -> _Geometry:
    ac = _airbag_cfg(cfg)
    hit = _GEOM_CACHE.get(id(model))
    if hit is None or hit[0] is not model or hit[1] != ac:
        hit = (model, ac, _Geometry(model, cfg))
        _GEOM_CACHE[id(model)] = hit
    return hit[2]


def set_airbags(model: mujoco.MjModel, deployed: bool | float, cfg: dict | None = None) -> None:
    """Stow (False / 0), deploy (True / 1) or partially inflate (fraction in (0, 1)) every
    airbag by geometry.  ``cfg`` supplies ``suit.airbags`` (default config if None)."""
    _geometry(model, cfg).apply(model, float(deployed))


def airbag_fraction(model: mujoco.MjModel, cfg: dict | None = None) -> float:
    """Current inflation fraction (0 stowed .. 1 fully deployed)."""
    return _geometry(model, cfg).fraction(model)


def airbags_deployed(model: mujoco.MjModel, cfg: dict | None = None) -> bool:
    """True when the airbags are fully inflated."""
    ids = airbag_geom_ids(model)
    return bool(ids) and airbag_fraction(model, cfg) >= 1.0 - 1e-9


class AirbagState:
    """Per-env inflation state: ``deploy()`` starts a linear inflation over
    ``suit.airbags.deploy_time``; call ``step(model, dt)`` every physics step."""

    def __init__(self, model: mujoco.MjModel, cfg: dict | None = None):
        self.cfg = cfg
        self.deploy_time = float(_airbag_cfg(cfg)["deploy_time"])
        self.fraction = 0.0
        self.deploying = False

    def stow(self, model: mujoco.MjModel) -> None:
        self.fraction = 0.0
        self.deploying = False
        set_airbags(model, 0.0, self.cfg)

    def deploy(self) -> None:
        self.deploying = True

    @property
    def triggered(self) -> bool:
        return self.deploying or self.fraction > 0.0

    def step(self, model: mujoco.MjModel, dt: float) -> None:
        if not self.deploying or self.fraction >= 1.0:
            return
        if self.deploy_time <= 0.0:
            self.fraction = 1.0
        else:
            self.fraction = min(1.0, self.fraction + dt / self.deploy_time)
        set_airbags(model, self.fraction, self.cfg)


_CFG_CACHE: dict = {}


def _default_cfg() -> dict:
    if not _CFG_CACHE:
        _CFG_CACHE.update(load_config())
    return _CFG_CACHE
