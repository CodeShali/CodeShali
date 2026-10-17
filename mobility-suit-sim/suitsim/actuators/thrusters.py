"""Thruster bank: first-order-lag thrusters with shared consumable budgets.

Model (per physics step ``dt``, vectorised over all N thrusters):

1. Command clamp: ``u = clip(f_cmd, 0, f_max_eff)`` (push only).
2. First-order lag, forward Euler (spec formula)::

       f += (u - f) * alpha,   alpha = min(dt / tau_eff, 1)

   The ``min(.,1)`` keeps the update unconditionally stable when a randomised tau drops
   below dt (alpha = 1 means "no lag"). With dt = 1 ms and tau = 5 ms the Euler response
   after one tau is 1 - (1 - 0.2)^5 = 67.2 % (vs. 63.2 % for the continuous system).
3. Output clamp to ``[0, f_max_eff]``.
4. Budget deduction:
   * gas budgets (``impulse_gas_pods`` / ``impulse_gas_jets``): impulse ``sum(f) * dt``
     from the class's shared tank;
   * ``battery_wh`` (fans): ``sum(k f^1.5) * dt / 3600`` with ``k = P_peak / f_max^1.5``
     fitted on the NOMINAL f_max from the config (domain randomisation does not change k).
   If a step would overdraw a budget, all forces on that budget are scaled uniformly so the
   budget lands exactly at 0 (impulse is linear in f -> s = rem/demand; fan energy goes
   like f^1.5 -> s = (rem/demand)^(2/3)). Once a budget is 0, every thruster on it outputs 0
   and its ``u_max`` is 0.
"""
from __future__ import annotations

import numpy as np

from suitsim.interfaces import BUDGET_KEYS, THRUSTER_CLASSES, ThrusterSpec

_SEC_PER_HOUR = 3600.0
_BATTERY_KEY = "battery_wh"


def fan_k(cfg: dict) -> float:
    """Fan power coefficient k [W / N^1.5] so that P(f_max) == fan_peak_power_w."""
    p_peak = float(cfg["suit"]["budgets"]["fan_peak_power_w"])
    f_max = float(cfg["suit"]["actuator_classes"]["fan"]["f_max"])
    return p_peak / f_max ** 1.5


def fan_power(f: np.ndarray | float, k: float | dict) -> np.ndarray | float:
    """Electrical power [W] of a fan producing thrust ``f`` [N]: ``P = k f^1.5``.

    ``k`` is either the coefficient itself or a config dict (then :func:`fan_k` is used).
    """
    if isinstance(k, dict):
        k = fan_k(k)
    f = np.maximum(np.asarray(f, dtype=float), 0.0)
    p = k * f ** 1.5
    return float(p) if p.ndim == 0 else p


class ThrusterBank:
    """Implements :class:`suitsim.interfaces.ThrusterBankProtocol`."""

    def __init__(self, specs: list[ThrusterSpec], cfg: dict, dt: float):
        self.specs = list(specs)
        self.n = len(self.specs)
        self.dt = float(dt)
        self.names = [s.name for s in self.specs]

        self.f_max_nom = np.array([s.f_max for s in self.specs], dtype=float)
        self.tau_nom = np.array([s.tau for s in self.specs], dtype=float)
        cls = np.array([s.cls for s in self.specs])
        self._cls_idx = {c: np.flatnonzero(cls == c) for c in THRUSTER_CLASSES}
        self.idx_pods = self._cls_idx["lift_pod"]
        self.idx_fans = self._cls_idx["fan"]
        self.idx_jets = self._cls_idx["micro_jet"]

        budgets = cfg["suit"]["budgets"]
        self.capacity = {k: float(budgets[k]) for k in BUDGET_KEYS}
        bud = np.array([s.budget for s in self.specs])
        unknown = set(bud) - set(BUDGET_KEYS)
        if unknown:
            raise ValueError(f"unknown budget keys in thruster specs: {unknown}")
        self._bud_idx = {k: np.flatnonzero(bud == k) for k in BUDGET_KEYS}
        self._gas_keys = tuple(k for k in BUDGET_KEYS if k != _BATTERY_KEY)
        self.k_fan = fan_k(cfg)

        self._bud_of = bud
        self.reset()

    # ------------------------------------------------------------------ setup
    def class_index(self, cls: str) -> np.ndarray:
        return self._cls_idx[cls]

    def budget_index(self, key: str) -> np.ndarray:
        return self._bud_idx[key]

    def reset(self, scale_fmax: np.ndarray | None = None,
              scale_tau: np.ndarray | None = None) -> None:
        """Zero forces, refill budgets, apply per-thruster multiplicative DR scales."""
        sf = np.ones(self.n) if scale_fmax is None else np.broadcast_to(
            np.asarray(scale_fmax, dtype=float), (self.n,))
        st = np.ones(self.n) if scale_tau is None else np.broadcast_to(
            np.asarray(scale_tau, dtype=float), (self.n,))
        if np.any(sf < 0) or np.any(st <= 0):
            raise ValueError("scale_fmax must be >= 0 and scale_tau > 0")
        self.f_max = self.f_max_nom * sf
        self.tau = self.tau_nom * st
        self.alpha = np.minimum(self.dt / self.tau, 1.0)
        self._f = np.zeros(self.n)
        self._cmd = np.zeros(self.n)
        self._rem = dict(self.capacity)
        self._alive = np.ones(self.n, dtype=bool)
        self._fan_power = np.zeros(len(self.idx_fans))

    # ------------------------------------------------------------------ step
    def step(self, f_cmd: np.ndarray) -> np.ndarray:
        f_cmd = np.asarray(f_cmd, dtype=float)
        if f_cmd.shape != (self.n,):
            raise ValueError(f"f_cmd must have shape ({self.n},), got {f_cmd.shape}")
        u_max = self.u_max
        cmd = np.clip(np.nan_to_num(f_cmd, nan=0.0), 0.0, u_max)
        self._cmd = cmd
        f = self._f + (cmd - self._f) * self.alpha
        np.clip(f, 0.0, u_max, out=f)

        dt = self.dt
        for key in self._gas_keys:
            idx = self._bud_idx[key]
            if idx.size == 0:
                continue
            demand = f[idx].sum() * dt
            rem = self._rem[key]
            if demand > rem:
                f[idx] *= rem / demand if demand > 0 else 0.0
                self._rem[key] = 0.0
                self._alive[idx] = False
            else:
                self._rem[key] = rem - demand
                if self._rem[key] <= 0.0:
                    self._alive[idx] = False

        idx = self._bud_idx[_BATTERY_KEY]
        if idx.size:
            p = self.k_fan * f[idx] ** 1.5
            demand = p.sum() * dt / _SEC_PER_HOUR
            rem = self._rem[_BATTERY_KEY]
            if demand > rem:
                s = (rem / demand) ** (2.0 / 3.0) if demand > 0 else 0.0
                f[idx] *= s
                p *= s ** 1.5
                self._rem[_BATTERY_KEY] = 0.0
                self._alive[idx] = False
            else:
                self._rem[_BATTERY_KEY] = rem - demand
                if self._rem[_BATTERY_KEY] <= 0.0:
                    self._alive[idx] = False
            # fan-power array is ordered like idx_fans (fans are the battery consumers)
            self._fan_power = self.k_fan * f[self.idx_fans] ** 1.5

        self._f = f
        return f.copy()

    # ------------------------------------------------------------------ queries
    @property
    def forces(self) -> np.ndarray:
        return self._f.copy()

    @property
    def commands(self) -> np.ndarray:
        return self._cmd.copy()

    @property
    def u_max(self) -> np.ndarray:
        return np.where(self._alive, self.f_max, 0.0)

    def remaining(self) -> dict[str, float]:
        return {k: float(self._rem[k]) for k in BUDGET_KEYS}

    def used(self) -> dict[str, float]:
        return {k: float(self.capacity[k] - self._rem[k]) for k in BUDGET_KEYS}

    def fan_power_w(self) -> np.ndarray:
        """(n_fans,) current electrical power per fan [W], order of idx_fans."""
        return self._fan_power.copy()

    def telemetry(self) -> dict:
        return {
            "forces": self._f.copy(),
            "commands": self._cmd.copy(),
            "u_max": self.u_max,
            "remaining": self.remaining(),
            "used": self.used(),
            "class_force": {c: float(self._f[i].sum()) for c, i in self._cls_idx.items()},
            "fan_power_w": self._fan_power.copy(),
        }
