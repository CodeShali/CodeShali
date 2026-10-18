"""Domain randomisation of the model: human segment mass/inertia and ground friction.

Airbag geoms out-prioritise the ground in contacts (their soft solref/solimp apply), so their
friction is kept equal to the ground's here.
"""
from __future__ import annotations

import mujoco
import numpy as np

from suitsim.interfaces import GROUND_GEOM, SEGMENTS, load_config
from suitsim.model.airbags import airbag_geom_ids


class ModelRandomizer:
    """Keeps the nominal values of ``model`` so repeated ``apply`` calls are idempotent.

    ``apply(mass_scale, friction)`` scales only the HUMAN part of each segment's mass and
    inertia (suit-mass geoms are untouched) and sets the ground sliding friction.
    """

    def __init__(self, model: mujoco.MjModel, cfg: dict | None = None):
        self.model = model
        self.body_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, s)
                                  for s in SEGMENTS], int)
        self.ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, GROUND_GEOM)
        self.nom_mass = model.body_mass.copy()
        self.nom_inertia = model.body_inertia.copy()
        self.nom_friction = model.geom_friction[self.ground].copy()
        self.airbags = np.asarray(airbag_geom_ids(model), int)   # friction mirrors the ground
        # suit (non-scaled) contribution per body: point-like spheres placed at the segment
        # geom centre == body CoM (see builder), so the parallel-axis term vanishes.
        cfg = cfg if cfg is not None else load_config()
        r = float(cfg["human"]["geometry"]["suit_mass_radius"])
        self.suit_mass = np.zeros(model.nbody)
        for spec in cfg["suit"]["masses"].values():
            share = float(spec["total"]) / len(spec["bodies"])
            for b in spec["bodies"]:
                self.suit_mass[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)] += share
        self.suit_inertia = np.repeat((0.4 * self.suit_mass * r ** 2)[:, None], 3, axis=1)
        self.mass_scale = 1.0

    def apply(self, mass_scale: float = 1.0, friction: float | None = None) -> None:
        m = self.model
        for b in self.body_ids:
            hm = self.nom_mass[b] - self.suit_mass[b]
            hi = self.nom_inertia[b] - self.suit_inertia[b]
            m.body_mass[b] = self.suit_mass[b] + mass_scale * hm
            m.body_inertia[b] = self.suit_inertia[b] + mass_scale * hi
        self.mass_scale = float(mass_scale)
        m.geom_friction[self.ground] = self.nom_friction
        if friction is not None:
            m.geom_friction[self.ground, 0] = float(friction)
        m.geom_friction[self.airbags] = m.geom_friction[self.ground]
        _set_const(m)

    def restore(self) -> None:
        self.apply(1.0, None)


def _set_const(model: mujoco.MjModel) -> None:
    """Recompute derived constants (subtree masses, invweights, actuator_acc0...)."""
    mujoco.mj_setConst(model, mujoco.MjData(model))


_RANDOMIZERS: dict[int, tuple[mujoco.MjModel, ModelRandomizer]] = {}


def _randomizer(model: mujoco.MjModel) -> ModelRandomizer:
    hit = _RANDOMIZERS.get(id(model))
    if hit is None or hit[0] is not model:
        hit = (model, ModelRandomizer(model))
        _RANDOMIZERS[id(model)] = hit
    return hit[1]


def apply_mass_scale(model: mujoco.MjModel, scale: float) -> None:
    """Idempotent: scales the human masses relative to the values first seen for ``model``."""
    r = _randomizer(model)
    fr = float(model.geom_friction[r.ground, 0])
    r.apply(scale, fr)


def set_ground_friction(model: mujoco.MjModel, mu: float) -> None:
    r = _randomizer(model)
    model.geom_friction[r.ground] = r.nom_friction
    model.geom_friction[r.ground, 0] = float(mu)
    model.geom_friction[r.airbags] = model.geom_friction[r.ground]
