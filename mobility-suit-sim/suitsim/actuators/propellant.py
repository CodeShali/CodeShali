"""Propellant mass depletion: the gas / charges a budget burns leave the suit.

Impulse ``I`` used from a budget with specific impulse ``Isp`` has consumed propellant mass
``m = I / (Isp * g0)`` (``g0 = 9.80665``).  The propellant of each budget is carried on one
segment (``suit.propellant.<budget>.body``, e.g. pods: pelvis, jet tank: torso) as a suit
point mass at that segment's CoM (see :func:`suitsim.interfaces.suit_mass_items` and the
builder), so depleting it only lowers the body's mass and its inertia share — the body CoM
(``xipos``) does not move.

MuJoCo semantics (verified, see tests/test_propulsion.py):

* ``model.body_mass`` / ``model.body_inertia`` may be changed between steps: the forward
  dynamics (``mj_comPos`` -> ``cinert`` -> mass matrix, gravity and bias forces) read them
  directly, so no ``mj_setConst`` is needed for the dynamics;
* but ``model.body_subtreemass`` is a compile-time constant (refreshed only by
  ``mj_setConst``) and ``subtree_com`` / ``mj_subtreeVel`` divide by it — so the tracker
  updates it too (the body and all its ancestors lose ``dm``).  ``body_invweight0`` /
  ``dof_invweight0`` (constraint-solver scaling only) are left at the full-load values: the
  propellant is ~1-2 % of the total mass.

The tracker composes with :class:`suitsim.model.ModelRandomizer`: the randomiser rewrites
every segment's mass from its stored nominal (full-tank) values and calls ``mj_setConst``,
then :meth:`PropellantTracker.capture` takes that state as "full"; :meth:`restore` puts it
back (episode reset).  Usage per step comes from the thruster bank's ``used()`` dict.
"""
from __future__ import annotations

import mujoco
import numpy as np

from suitsim.interfaces import G0, PropellantSpec, propellant_specs


class PropellantTracker:
    """Deplete MuJoCo body masses as gas budgets are used.

    Parameters
    ----------
    model : the MuJoCo model (modified in place).
    specs : propellant per budget (``propellant_specs(cfg)``).
    suit_mass_radius : radius [m] of the point-like suit-mass spheres (their own inertia
        ``0.4 m r^2`` is removed in proportion to the burnt propellant).
    """

    def __init__(self, model: mujoco.MjModel, specs: list[PropellantSpec],
                 suit_mass_radius: float = 0.0):
        self.model = model
        self.specs = list(specs)
        self.body_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, p.body)
                                  for p in self.specs], int)
        if np.any(self.body_ids < 0):
            raise KeyError(f"propellant bodies missing from model: "
                           f"{[p.body for p, b in zip(self.specs, self.body_ids) if b < 0]}")
        self.c = np.array([p.isp * G0 for p in self.specs])           # exhaust velocity [m/s]
        self.m0 = np.array([p.mass for p in self.specs])               # full load [kg]
        self.r2 = 0.4 * float(suit_mass_radius) ** 2                   # sphere inertia / mass
        # ancestor chains (body itself first, world excluded) for the subtree-mass update
        self.chains = []
        for b in self.body_ids:
            chain = []
            while b > 0:
                chain.append(int(b))
                b = int(model.body_parentid[b])
            self.chains.append(np.array(chain, int))
        self.used_kg = np.zeros(len(self.specs))
        self.capture()

    @classmethod
    def from_config(cls, model: mujoco.MjModel, cfg: dict) -> "PropellantTracker":
        return cls(model, propellant_specs(cfg),
                   float(cfg["human"]["geometry"].get("suit_mass_radius", 0.0)))

    # ------------------------------------------------------------------ state
    def capture(self) -> None:
        """Take the model's current mass state as the full-load reference (call after the
        domain randomiser has written the episode's masses)."""
        m = self.model
        self._mass = m.body_mass.copy()
        self._inertia = m.body_inertia.copy()
        self._subtree = m.body_subtreemass.copy()
        self.used_kg[:] = 0.0

    def restore(self) -> None:
        """Write the full-load reference back (refill every tank)."""
        m = self.model
        m.body_mass[:] = self._mass
        m.body_inertia[:] = self._inertia
        m.body_subtreemass[:] = self._subtree
        self.used_kg[:] = 0.0

    # ------------------------------------------------------------------ update
    def update(self, used_impulse: dict[str, float]) -> None:
        """Set the masses for the total impulse used so far per budget [N*s]."""
        used = np.array([float(used_impulse.get(p.budget, 0.0)) for p in self.specs])
        kg = np.clip(used / self.c, 0.0, self.m0)
        if np.array_equal(kg, self.used_kg):
            return
        self.used_kg = kg
        m = self.model
        m.body_mass[self.body_ids] = self._mass[self.body_ids]
        m.body_inertia[self.body_ids] = self._inertia[self.body_ids]
        sub = self._subtree.copy()
        for b, chain, dm in zip(self.body_ids, self.chains, kg):
            m.body_mass[b] -= dm
            m.body_inertia[b] -= self.r2 * dm
            sub[chain] -= dm
        sub[0] = self._subtree[0] - kg.sum()
        m.body_subtreemass[:] = sub

    def used(self) -> dict[str, float]:
        """Propellant mass burnt per budget [kg]."""
        return {p.budget: float(k) for p, k in zip(self.specs, self.used_kg)}

    def remaining(self) -> dict[str, float]:
        return {p.budget: float(m0 - k) for p, m0, k in zip(self.specs, self.m0, self.used_kg)}
