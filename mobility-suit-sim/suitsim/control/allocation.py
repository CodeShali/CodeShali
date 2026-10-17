"""Thrust allocation (spec 5.1, milestone M3).

Maps a desired wrench about the CoM ``w = [Fx, Fy, Fz, Tx, Ty, Tz]`` (world-aligned) onto
non-negative thruster force magnitudes ``u`` (N,), order of :func:`load_thruster_specs`.

Problem solved every control step
---------------------------------
With ``A`` the (6, N) allocation matrix, column ``i = [d_i ; r_i x d_i]`` (``r_i`` thruster
position relative to the CoM, ``d_i`` unit force direction, both world-aligned and supplied
by the caller because the limbs move), and the change of variables to *normalised commands*
``v_i = u_i / s_i`` where ``s_i`` is the thruster's NOMINAL ``f_max`` from the YAML (not the
currently available ``u_max``, so the class preference does not drift as budgets deplete)::

    min_v  || W (A S v - w) ||^2  +  rho * sum_i lambda_i v_i^2
    s.t.   0 <= v_i <= u_max_i / s_i,                 S = diag(s)

* ``W = diag(1, 1, 1, 1/L, 1/L, 1/L)`` — wrench row weighting; ``L`` is a characteristic
  length (``allocation.torque_length``, default 0.3 m) so a torque error of ``L`` N·m costs as
  much as a 1 N force error.
* ``lambda_i`` = ``suit.actuator_classes[cls].alloc_lambda`` — dimensionless weight on the
  squared *normalised* command.  Per Newton² the penalty is ``lambda_i / s_i^2``
  (lift pod 4.4e-9, micro jet 6.3e-8, fan 3.3e-7 with the default YAML), so in the null space of
  ``A`` large vertical force goes to the pods, torque to the (well-levered) micro jets.
  ``allocation.class_lambda: {cls: value}`` optionally overrides the class weights.
* ``rho`` = ``allocation.lambda_scale`` (default 1.0).  Because the tracking term is in N²
  and a full-scale command costs only ``lambda`` (<= 0.03), tracking dominates: the
  regularisation bias on a feasible wrench is << 0.5 %, while redundancy is still resolved by
  the class weights.
* Stage 2 (``allocation.refine``): if the relative residual of stage 1 exceeds
  ``allocation.refine_tol`` (default 0.5 %), the problem is re-solved with ``rho`` multiplied
  by ``allocation.refine_factor`` (default 1e-3) and the better solution is kept.  It is
  skipped above ``allocation.refine_max`` (default 10 %), where the request is clearly
  infeasible and the tiny lambda bias is irrelevant.  In practice it only triggers for
  marginally feasible / saturated requests (never in the M3 random set).

Implementation: the regulariser is stacked under the tracking rows,
``[W A S ; sqrt(rho Lambda)] v ~= [W w ; 0]``, and solved with ``scipy.optimize.lsq_linear``
(method ``allocation.method``, default ``'bvls'``).  Thrusters whose ``u_max`` is ~0
(exhausted budget / disabled) are removed from the problem and get ``u = 0``.  Infeasible
requests return the bounded least-squares best effort — never an exception.

``last_residual`` is ``||W (A u - w)|| / max(||W w||, eps)`` for the last call.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear

from suitsim.interfaces import ThrusterSpec

__all__ = ["Allocator", "nominal_geometry", "skeleton_origins", "axis_authority"]

_DEFAULTS = {
    "torque_length": 0.3,      # [m] characteristic length for torque-row weighting
    "lambda_scale": 1.0,       # rho, global multiplier on class alloc_lambda
    "method": "bvls",          # lsq_linear method ('bvls' | 'trf')
    "lsq_tol": 1e-10,          # lsq_linear tol
    "max_iter": None,          # lsq_linear max_iter (None = solver default)
    "refine": True,            # stage-2 re-solve when the residual is large
    "refine_tol": 5e-3,        # relative residual that triggers stage 2
    "refine_factor": 1e-3,     # rho multiplier for stage 2
    "refine_max": 0.1,         # skip stage 2 above this residual (clearly infeasible: the
                               # lambda bias cannot explain it, re-solving only costs time)
    "umax_eps": 1e-9,          # [N] u_max below this => thruster excluded
    "class_lambda": {},        # optional {cls: lambda} override of actuator_classes.alloc_lambda
}


class Allocator:
    """Bounded, class-weighted least-squares thrust allocator (implements AllocatorProtocol)."""

    def __init__(self, specs: list[ThrusterSpec], cfg: dict | None = None, **overrides):
        self.specs = list(specs)
        self.n = len(self.specs)
        acfg = dict(_DEFAULTS)
        acfg.update((cfg or {}).get("allocation", {}) or {})
        acfg.update(overrides)
        self.params = acfg
        self.scale = np.array([s.f_max for s in self.specs], dtype=float)        # s_i
        cl = acfg.get("class_lambda") or {}
        self.lam = np.array([float(cl.get(s.cls, s.alloc_lambda)) for s in self.specs])  # lambda_i
        L = float(acfg["torque_length"])
        self.W = np.array([1.0, 1.0, 1.0, 1.0 / L, 1.0 / L, 1.0 / L])
        self.method = str(acfg["method"])
        self.last_residual = 0.0
        self.last_stage = 0
        self.last_status = None

    # ------------------------------------------------------------------ geometry
    def matrix(self, r: np.ndarray, d: np.ndarray) -> np.ndarray:
        """(6, N) allocation matrix, column i = [d_i ; r_i x d_i]."""
        r = np.asarray(r, dtype=float).reshape(self.n, 3)
        d = np.asarray(d, dtype=float).reshape(self.n, 3)
        A = np.empty((6, self.n))
        A[:3] = d.T
        A[3:] = np.cross(r, d).T
        return A

    def residual(self, A: np.ndarray, u: np.ndarray, wrench: np.ndarray) -> float:
        """Relative weighted residual ||W(Au - w)|| / max(||W w||, eps)."""
        w = np.asarray(wrench, dtype=float)
        den = max(np.linalg.norm(self.W * w), 1e-12)
        return float(np.linalg.norm(self.W * (A @ u - w)) / den)

    # ------------------------------------------------------------------ solve
    def _solve(self, M: np.ndarray, b: np.ndarray, sq_lam: np.ndarray, ub: np.ndarray):
        k = M.shape[1]
        aug = np.empty((6 + k, k))
        aug[:6] = M
        aug[6:] = np.diag(sq_lam)
        rhs = np.zeros(6 + k)
        rhs[:6] = b
        res = lsq_linear(aug, rhs, bounds=(np.zeros(k), ub), method=self.method,
                         tol=self.params["lsq_tol"], max_iter=self.params["max_iter"])
        return np.clip(res.x, 0.0, ub), res.status

    def allocate(self, wrench: np.ndarray, r: np.ndarray, d: np.ndarray,
                 u_max: np.ndarray | None = None) -> np.ndarray:
        """Thruster forces u (N,), 0 <= u <= u_max, approximately reproducing ``wrench``."""
        w = np.asarray(wrench, dtype=float).reshape(6)
        A = self.matrix(r, d)
        u_max = self.scale.copy() if u_max is None else np.asarray(u_max, dtype=float)
        u = np.zeros(self.n)
        active = u_max > self.params["umax_eps"]
        if not active.any() or not np.any(w):
            self.last_residual = self.residual(A, u, w) if np.any(w) else 0.0
            self.last_stage = 0
            return u
        s = self.scale[active]
        M = (self.W[:, None] * A[:, active]) * s[None, :]
        b = self.W * w
        ub = u_max[active] / s
        rho = float(self.params["lambda_scale"])
        v, status = self._solve(M, b, np.sqrt(rho * self.lam[active]), ub)
        u[active] = v * s
        res = self.residual(A, u, w)
        stage = 1
        if self.params["refine"] and self.params["refine_tol"] < res <= self.params["refine_max"]:
            rho2 = rho * float(self.params["refine_factor"])
            v2, status2 = self._solve(M, b, np.sqrt(rho2 * self.lam[active]), ub)
            u2 = np.zeros(self.n)
            u2[active] = v2 * s
            res2 = self.residual(A, u2, w)
            if res2 < res:
                u, res, status, stage = u2, res2, status2, 2
        self.last_residual = res
        self.last_stage = stage
        self.last_status = status
        return u


# ---------------------------------------------------------------------- test helpers
def skeleton_origins(cfg: dict) -> dict[str, np.ndarray]:
    """World positions of every body origin in the nominal standing pose (all body frames
    world-aligned), accumulated along ``human.skeleton`` parent chains."""
    sk = cfg["human"]["skeleton"]
    out: dict[str, np.ndarray] = {}

    def origin(name: str) -> np.ndarray:
        if name == "world":
            return np.zeros(3)
        if name not in out:
            out[name] = origin(sk[name]["parent"]) + np.asarray(sk[name]["pos"], float)
        return out[name]

    for name in sk:
        origin(name)
    return out


def nominal_geometry(specs: list[ThrusterSpec], cfg: dict,
                     com: np.ndarray | None = None,
                     rotation: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """FOR TESTS ONLY (the estimator provides real geometry).  World-aligned thruster
    positions relative to the CoM and unit directions in the nominal standing pose, optionally
    rotated as a rigid whole by ``rotation`` (3x3).  ``com`` defaults to pelvis + [0,0,0.1]."""
    org = skeleton_origins(cfg)
    if com is None:
        com = org["pelvis"] + np.array([0.0, 0.0, 0.1])
    r = np.array([org[s.body] + s.pos for s in specs]) - com
    d = np.array([s.dir for s in specs])
    if rotation is not None:
        R = np.asarray(rotation, float)
        r, d = r @ R.T, d @ R.T
    return r, d


def axis_authority(A: np.ndarray, u_max: np.ndarray, axis: int,
                   base: np.ndarray | None = None) -> tuple[float, float]:
    """(min, max) of wrench component ``axis`` achievable with 0 <= u <= u_max while every
    other component equals ``base`` (default 0), via ``scipy.optimize.linprog``.
    Returns (nan, nan) if ``base`` itself is infeasible."""
    from scipy.optimize import linprog
    base = np.zeros(6) if base is None else np.asarray(base, float)
    others = [k for k in range(6) if k != axis]
    out = []
    for sign in (1.0, -1.0):
        res = linprog(sign * A[axis], A_eq=A[others], b_eq=base[others],
                      bounds=list(zip(np.zeros(A.shape[1]), u_max)), method="highs")
        out.append(float(A[axis] @ res.x) if res.status == 0 else float("nan"))
    return out[0], out[1]
