"""Aggregate per-episode ground-truth ``info`` dicts into evaluation summaries.

Every env returns :data:`suitsim.interfaces.EPISODE_INFO_KEYS` in ``info`` at episode end.
:func:`summarize` turns a list of those dicts (optionally with ``seed`` / ``episode_params``)
into one JSON-serialisable summary: rates with Wilson 95 % CIs, distribution statistics,
budget usage / compliance (T4) and a failure-mode breakdown.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from suitsim.interfaces import BUDGET_KEYS, EPISODE_INFO_KEYS, budget_capacity

# Defaults mirror configs/task_drop.yaml::safe_landing
DEFAULT_SAFE_LIMITS = {"max_vertical_speed": 2.0, "max_horizontal_speed": 1.0,
                       "max_tilt_deg": 20.0}
DEFAULT_BUDGETS = {"impulse_gas_pods": 2500.0, "impulse_gas_jets": 600.0, "battery_wh": 150.0}

FAILURE_MODES: tuple[str, ...] = (
    "bad_contact",        # head/torso/pelvis touched ground before the feet
    "timeout",            # never touched down (first_contact == 'none')
    "vertical_too_fast",  # |impact_v_vertical| > max_vertical_speed
    "horizontal_too_fast",
    "tilt",               # tilt_at_impact_deg > max_tilt_deg
    "task_fail",          # landing was safe but the task-level goal was not met
    "other",              # unsafe, but no criterion above explains it
)

CONTINUOUS_KEYS: tuple[str, ...] = (
    "impact_v_vertical", "impact_v_horizontal", "tilt_at_impact_deg",
    "impulse_gas_pods", "impulse_gas_jets", "battery_wh", "episode_time",
)
_ABS_KEYS = ("impact_v_vertical", "impact_v_horizontal")
# reported only when the episodes carry them (hardware-realistic propulsion options)
OPTIONAL_KEYS: tuple[str, ...] = (
    "propellant_kg", "propellant_kg_pods", "propellant_kg_jets", "total_mass_end",
    "pulse_charges_used", "pulse_first_ignition_t", "valve_pulses",
)


# --------------------------------------------------------------------------------------
# Small statistics helpers
# --------------------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n (default 95 %)."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if k <= 0 else max(0.0, centre - half)
    hi = 1.0 if k >= n else min(1.0, centre + half)
    return (lo, hi)


def rate(flags: Sequence[bool]) -> dict:
    flags = [bool(f) for f in flags]
    n, k = len(flags), int(sum(flags))
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else float("nan"), "ci95": [lo, hi]}


def describe(values: Iterable[float]) -> dict:
    """mean / std / min / p50 / p95 / max of the finite values (NaNs ignored)."""
    v = np.asarray([x for x in values if x is not None], dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "p50": None, "p95": None,
                "max": None}
    return {"n": int(v.size), "mean": float(v.mean()), "std": float(v.std()),
            "min": float(v.min()), "p50": float(np.percentile(v, 50)),
            "p95": float(np.percentile(v, 95)), "max": float(v.max())}


def _f(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------------------
# Config helpers
# --------------------------------------------------------------------------------------
def safe_limits_from_cfg(cfg: dict | None) -> dict:
    out = dict(DEFAULT_SAFE_LIMITS)
    out.update({k: float(v) for k, v in ((cfg or {}).get("safe_landing") or {}).items()
                if k in out})
    return out


def budgets_from_cfg(cfg: dict | None) -> dict:
    b = ((cfg or {}).get("suit") or {}).get("budgets") or {}
    out = dict(DEFAULT_BUDGETS)
    out.update({k: float(b[k]) for k in BUDGET_KEYS if k in b})
    suit = (cfg or {}).get("suit") or {}
    if suit.get("thrusters") and suit.get("actuator_classes"):
        # a solid-pulse budget holds exactly its charges (interfaces.budget_capacity)
        out.update({k: budget_capacity(cfg, k) for k in BUDGET_KEYS if k in b})
    return out


# --------------------------------------------------------------------------------------
# Per-episode classification
# --------------------------------------------------------------------------------------
def failure_flags(info: dict, limits: dict | None = None) -> dict[str, bool]:
    """Multi-label failure criteria of one episode (all False for a successful one)."""
    lim = {**DEFAULT_SAFE_LIMITS, **(limits or {})}
    first = str(info.get("first_contact", "none"))
    no_contact = first == "none"
    vv = abs(_f(info.get("impact_v_vertical"), 0.0))
    vh = abs(_f(info.get("impact_v_horizontal"), 0.0))
    tilt = abs(_f(info.get("tilt_at_impact_deg"), 0.0))
    flags = {
        "bad_contact": bool(info.get("bad_contact", False)),
        "timeout": no_contact,
        "vertical_too_fast": (not no_contact) and vv > lim["max_vertical_speed"],
        "horizontal_too_fast": (not no_contact) and vh > lim["max_horizontal_speed"],
        "tilt": (not no_contact) and tilt > lim["max_tilt_deg"],
    }
    safe = bool(info.get("safe", False))
    success = bool(info.get("success", safe))
    if success:
        return {m: False for m in FAILURE_MODES}
    flags["task_fail"] = safe and not success
    flags["other"] = (not safe) and not any(flags.values())
    return flags


def primary_failure(info: dict, limits: dict | None = None) -> str:
    """Single failure label by priority (FAILURE_MODES order); 'none' on success."""
    flags = failure_flags(info, limits)
    for m in FAILURE_MODES:
        if flags.get(m):
            return m
    return "none"


def outcome_label(info: dict, limits: dict | None = None) -> str:
    """Human-readable outcome for titles: 'success' or 'failure: <mode>' (never a bare
    'none', which reads like a controller name)."""
    m = primary_failure(info, limits)
    return "success" if m == "none" else f"failure: {m}"


def episode_badness(info: dict, limits: dict | None = None) -> float:
    """Continuous 'how bad was it' score (0 = perfect landing, dimensionless)."""
    lim = {**DEFAULT_SAFE_LIMITS, **(limits or {})}
    if str(info.get("first_contact", "none")) == "none":
        return 100.0
    s = (abs(_f(info.get("impact_v_vertical"), 0.0)) / lim["max_vertical_speed"]
         + abs(_f(info.get("impact_v_horizontal"), 0.0)) / lim["max_horizontal_speed"]
         + abs(_f(info.get("tilt_at_impact_deg"), 0.0)) / lim["max_tilt_deg"])
    if info.get("bad_contact"):
        s += 10.0
    return float(s)


def episode_score(info: dict, limits: dict | None = None) -> tuple:
    """Sort key, higher = better: success first, then safe, then lower badness."""
    return (int(bool(info.get("success", False))), int(bool(info.get("safe", False))),
            -episode_badness(info, limits))


def worst_best(results: Sequence[dict], k: int = 5,
               key: Callable[[dict], Any] | None = None,
               limits: dict | None = None) -> tuple[list[dict], list[dict]]:
    """Return (worst k, best k) episodes. ``key`` maps a result -> sortable, higher=better.

    Ties are broken by seed (stable). Worst list starts with the very worst; best list
    starts with the very best. The two lists never share an episode when len >= 2k.
    """
    key = key or (lambda r: episode_score(r, limits))
    idx = sorted(range(len(results)),
                 key=lambda i: (key(results[i]), -_f(results[i].get("seed"), i)))
    k = max(0, min(k, len(results)))
    worst = [results[i] for i in idx[:k]]
    best = [results[i] for i in list(reversed(idx))[:k]]
    return worst, best


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------
def summarize(results: Sequence[dict], cfg: dict | None = None,
              budgets: dict | None = None, limits: dict | None = None) -> dict:
    """Aggregate episode info dicts into a summary dict (JSON-serialisable)."""
    limits = limits or safe_limits_from_cfg(cfg)
    budgets = budgets or budgets_from_cfg(cfg)
    n = len(results)
    out: dict[str, Any] = {
        "n": n,
        "task": ((cfg or {}).get("task") or {}).get("name"),
        "limits": limits,
        "budgets": budgets,
        "success": rate([r.get("success", r.get("safe", False)) for r in results]),
        "safe": rate([r.get("safe", False) for r in results]),
        "bad_contact": rate([r.get("bad_contact", False) for r in results]),
        "airbags_deployed": rate([r.get("airbags_deployed", False) for r in results]),
        "touched_down": rate([str(r.get("first_contact", "none")) != "none"
                              for r in results]),
    }
    missing = sorted({k for r in results for k in EPISODE_INFO_KEYS if k not in r})
    if missing:
        out["missing_info_keys"] = missing

    stats = {}
    for key in CONTINUOUS_KEYS:
        vals = [_f(r.get(key)) for r in results]
        if key in _ABS_KEYS:
            vals = [abs(v) for v in vals]
        stats[key] = describe(vals)
    for key in OPTIONAL_KEYS:
        if any(key in r for r in results):
            stats[key] = describe(_f(r.get(key)) for r in results)
    # landed-only impact stats (timeouts have no meaningful impact speed)
    landed = [r for r in results if str(r.get("first_contact", "none")) != "none"]
    for key in ("impact_v_vertical", "impact_v_horizontal", "tilt_at_impact_deg"):
        stats[key + "_landed"] = describe(abs(_f(r.get(key))) for r in landed)
    out["stats"] = stats

    # Budget usage / compliance (T4)
    bud = {}
    all_within = np.ones(n, dtype=bool)
    for bk, cap in budgets.items():
        used = np.array([_f(r.get(bk), 0.0) for r in results], dtype=float)
        within = used <= cap * (1 + 1e-9) + 1e-9
        all_within &= within
        frac = used / cap if cap > 0 else np.full(n, np.nan)
        bud[bk] = {"budget": cap, "used": describe(used), "used_frac": describe(frac),
                   "within": rate(within.tolist())}
    bud["all_within"] = rate(all_within.tolist())
    out["budget_usage"] = bud

    # Failure modes
    primary: dict[str, int] = {m: 0 for m in FAILURE_MODES}
    multi: dict[str, int] = {m: 0 for m in FAILURE_MODES}
    first_contact: dict[str, int] = {}
    for r in results:
        p = primary_failure(r, limits)
        if p != "none":
            primary[p] += 1
        for m, f in failure_flags(r, limits).items():
            multi[m] += int(bool(f))
        fc = str(r.get("first_contact", "none"))
        first_contact[fc] = first_contact.get(fc, 0) + 1
    n_fail = n - out["success"]["k"]
    out["failures"] = {
        "n_fail": n_fail,
        "primary": primary,
        "primary_frac": {m: (c / n if n else 0.0) for m, c in primary.items()},
        "any": multi,
        "first_contact": dict(sorted(first_contact.items(), key=lambda kv: -kv[1])),
    }

    # Episode parameter ranges (for context)
    params: dict[str, list[float]] = {}
    for r in results:
        for pk, pv in (r.get("episode_params") or {}).items():
            if isinstance(pv, (int, float, np.floating, np.integer)) and not isinstance(pv, bool):
                params.setdefault(pk, []).append(float(pv))
    if params:
        out["episode_params"] = {k: describe(v) for k, v in params.items()}
    return out


def success_by_param(results: Sequence[dict], param: str, bins: int | Sequence[float] = 5
                     ) -> list[dict]:
    """Success rate (with CI) binned by one ``episode_params`` value, e.g. drop_height."""
    pairs = [(float(r["episode_params"][param]), bool(r.get("success", False)))
             for r in results if param in (r.get("episode_params") or {})]
    if not pairs:
        return []
    x = np.array([p[0] for p in pairs])
    s = np.array([p[1] for p in pairs])
    edges = np.linspace(x.min(), x.max() + 1e-9, bins + 1) if isinstance(bins, int) \
        else np.asarray(bins, dtype=float)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.any():
            rows.append({"lo": float(lo), "hi": float(hi), **rate(s[m].tolist())})
    return rows
