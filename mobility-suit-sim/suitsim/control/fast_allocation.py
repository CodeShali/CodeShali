"""Warm-started allocator: same problem as :class:`suitsim.control.allocation.Allocator`,
faster solver.

The allocation problem (stage 1 and stage 2 alike) is the strictly convex box QP::

    min_v  ||M v - b||^2 + sum_i lam_i v_i^2      s.t.  0 <= v <= ub

i.e. ``min 1/2 v'Hv - g'v`` with ``H = M'M + diag(lam)`` (positive definite since lam > 0)
and ``g = M'b``.  Its KKT point is unique, so any exact method returns the bvls answer.
This class replaces ``Allocator._solve`` by a primal (feasible) active-set method
(Nocedal & Wright, alg. 16.3, specialised to bounds) warm-started from the previous control
step's solution and working set — consecutive wrenches differ little at 500 Hz, so it
typically converges in a few linear solves of a <=44x44 system.  Strict convexity
guarantees finite termination; if the iteration cap is hit anyway the base bvls solver is
used, so results are never worse.

Everything else (row weighting, class lambdas, stage-2 refine, bookkeeping) is inherited.
"""
from __future__ import annotations

import numpy as np

from suitsim.control.allocation import Allocator

_PDAS_DEFAULTS = {"pdas_max_iter": 25, "pdas_kkt_tol": 1e-13}


class FastAllocator(Allocator):
    def __init__(self, specs, cfg=None, **overrides):
        super().__init__(specs, cfg, **overrides)
        p = dict(_PDAS_DEFAULTS)
        p.update({k: v for k, v in self.params.items() if k in _PDAS_DEFAULTS})
        self.pdas = p
        self._warm: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.n_fallback = 0
        self.n_solve = 0

    def reset_warm_start(self) -> None:
        self._warm.clear()

    def _solve(self, M: np.ndarray, b: np.ndarray, sq_lam: np.ndarray, ub: np.ndarray):
        """Primal active-set method for the box QP, warm-started from the last solution."""
        self.n_solve += 1
        k = M.shape[1]
        H = M.T @ M
        H[np.diag_indices(k)] += sq_lam * sq_lam
        g = M.T @ b
        key = (k, float(sq_lam @ sq_lam))
        warm = self._warm.get(key)
        # state: 0 free, -1 at lower bound (0), +1 at upper bound (ub)
        if warm is not None:
            st = warm[0].copy()
            v = np.clip(warm[1], 0.0, ub)
            st[(st == 1)] = 1
        else:
            st = -np.ones(k, int)
            v = np.zeros(k)
        v[st == -1] = 0.0
        v[st == 1] = ub[st == 1]
        tol = float(self.pdas["pdas_kkt_tol"]) * max(1.0, float(np.abs(g).max()))
        ok = False
        for _ in range(int(self.pdas["pdas_max_iter"]) + 2 * k):
            free = st == 0
            fixed = ~free
            vt = v.copy()
            vt[st == -1] = 0.0
            vt[st == 1] = ub[st == 1]
            if free.any():
                rhs = g[free] - H[np.ix_(free, fixed)] @ vt[fixed]
                try:
                    vt[free] = np.linalg.solve(H[np.ix_(free, free)], rhs)
                except np.linalg.LinAlgError:
                    break
            # blocking bounds along v -> vt
            dv = vt - v
            alpha, block, bside = 1.0, -1, 0
            if free.any():
                fi = np.flatnonzero(free)
                d = dv[fi]
                with np.errstate(divide="ignore", invalid="ignore"):
                    a_lo = np.where(d < -1e-15, -v[fi] / d, np.inf)
                    a_up = np.where(d > 1e-15, (ub[fi] - v[fi]) / d, np.inf)
                j_lo, j_up = int(np.argmin(a_lo)), int(np.argmin(a_up))
                if a_lo[j_lo] < alpha:
                    alpha, block, bside = float(a_lo[j_lo]), int(fi[j_lo]), -1
                if a_up[j_up] < alpha:
                    alpha, block, bside = float(a_up[j_up]), int(fi[j_up]), 1
            if block >= 0:
                v = v + max(alpha, 0.0) * dv
                st[block] = bside
                v[block] = 0.0 if bside == -1 else ub[block]
                continue
            v = vt
            mu = H @ v - g                     # >= 0 at lower, <= 0 at upper for optimality
            viol = np.where(st == -1, -mu, np.where(st == 1, mu, 0.0))
            j = int(np.argmax(viol))
            if viol[j] <= tol:
                ok = True
                break
            st[j] = 0
        if not ok:
            self.n_fallback += 1
            v, status = super()._solve(M, b, sq_lam, ub)
            st = np.where(v <= 1e-12, -1, np.where(v >= ub - 1e-12, 1, 0))
            self._warm[key] = (st, v.copy())
            return v, status
        v = np.clip(v, 0.0, ub)
        self._warm[key] = (st, v.copy())
        return v, 0
