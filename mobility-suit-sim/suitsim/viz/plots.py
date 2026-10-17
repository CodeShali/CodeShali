"""Time-series plots of one episode from its trajectory log.

The env provides ``info['trajectory']`` when built with ``log_trajectory=True``: a list of
dicts with (any subset of) keys ``t, com_height, com_vel (3,), tilt [rad], forces_by_class
{cls: N}, budgets {key: remaining}, phase``. Missing keys simply drop their panel.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from suitsim.interfaces import BUDGET_KEYS, THRUSTER_CLASSES  # noqa: E402


def _series(traj: Sequence[dict], key: str):
    vals = [p.get(key) for p in traj]
    if all(v is None for v in vals):
        return None
    return vals


def trajectory_arrays(traj: Sequence[dict]) -> dict[str, np.ndarray]:
    """Convert the list-of-dicts log to arrays (only keys that are present)."""
    out: dict[str, np.ndarray] = {}
    if not traj:
        return out
    t = _series(traj, "t")
    out["t"] = np.asarray([np.nan if v is None else v for v in t], dtype=float) if t \
        else np.arange(len(traj), dtype=float)
    for key in ("com_height", "feet_height", "tilt"):
        s = _series(traj, key)
        if s is not None:
            out[key] = np.asarray([np.nan if v is None else float(v) for v in s])
    s = _series(traj, "com_vel")
    if s is not None:
        v = np.asarray([[np.nan] * 3 if x is None else list(np.ravel(x))[:3] for x in s],
                       dtype=float)
        out["vz"] = v[:, 2]
        out["vh"] = np.linalg.norm(v[:, :2], axis=1)
    for key, names in (("forces_by_class", THRUSTER_CLASSES), ("budgets", BUDGET_KEYS)):
        s = _series(traj, key)
        if s is None:
            continue
        keys = list(names) + sorted({k for d in s if d for k in d} - set(names))
        for k in keys:
            arr = np.asarray([np.nan if not d or d.get(k) is None else float(d[k]) for d in s])
            if np.isfinite(arr).any():
                out[f"{key}.{k}"] = arr
    return out


def plot_episode(traj: Sequence[dict] | None, out_path: str | Path, info: dict | None = None,
                 cfg: dict | None = None, title: str | None = None) -> Path:
    """Multi-panel time-series figure (height, vertical/horizontal speed, tilt, thrust per
    class, remaining budgets). Writes a PNG even when the trajectory is missing."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    a = trajectory_arrays(traj or [])
    info = info or {}
    safe_cfg = (cfg or {}).get("safe_landing") or {}
    budgets_cap = ((cfg or {}).get("suit") or {}).get("budgets") or {}

    panels = []
    if "com_height" in a or "feet_height" in a:
        panels.append("height")
    if "vz" in a:
        panels.append("vel")
    if "tilt" in a:
        panels.append("tilt")
    if any(k.startswith("forces_by_class.") for k in a):
        panels.append("force")
    if any(k.startswith("budgets.") for k in a):
        panels.append("budget")

    if not panels:
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.axis("off")
        ax.text(0.5, 0.5, "no trajectory logged\n(construct env with log_trajectory=True)",
                ha="center", va="center")
        _annotate(fig, info, title)
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        return out_path

    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 2.0 * len(panels) + 0.8),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    t = a["t"]
    contact_t = _contact_time(traj, info)
    for ax, p in zip(axes, panels):
        if p == "height":
            if "com_height" in a:
                ax.plot(t, a["com_height"], label="CoM height")
            if "feet_height" in a:
                ax.plot(t, a["feet_height"], label="feet height")
            ax.axhline(0, color="0.6", lw=0.8)
            ax.set_ylabel("height [m]")
        elif p == "vel":
            ax.plot(t, a["vz"], label="vertical v")
            ax.plot(t, a["vh"], label="horizontal |v|")
            lim = safe_cfg.get("max_vertical_speed")
            if lim:
                ax.axhline(-float(lim), color="C3", ls="--", lw=1, label="-max vertical")
            ax.set_ylabel("velocity [m/s]")
        elif p == "tilt":
            ax.plot(t, np.degrees(a["tilt"]), color="C2", label="tilt")
            lim = safe_cfg.get("max_tilt_deg")
            if lim:
                ax.axhline(float(lim), color="C3", ls="--", lw=1, label="max tilt")
            ax.set_ylabel("tilt [deg]")
        elif p == "force":
            for k in sorted(x for x in a if x.startswith("forces_by_class.")):
                ax.plot(t, a[k], label=k.split(".", 1)[1])
            ax.set_ylabel("thrust [N]")
        elif p == "budget":
            for k in sorted(x for x in a if x.startswith("budgets.")):
                name = k.split(".", 1)[1]
                cap = budgets_cap.get(name)
                y = a[k] / float(cap) * 100 if cap else a[k]
                ax.plot(t, y, label=name)
            ax.set_ylabel("remaining [%]" if budgets_cap else "remaining")
        if contact_t is not None:
            ax.axvline(contact_t, color="k", ls=":", lw=1)
        ax.legend(fontsize=7, frameon=False, loc="best")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("time [s]")
    _annotate(fig, info, title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def _contact_time(traj, info) -> float | None:
    if info.get("contact_time") is not None:
        return float(info["contact_time"])
    return None


def _annotate(fig, info: dict, title: str | None):
    parts = []
    if title:
        parts.append(title)
    if info:
        ok = info.get("success")
        parts.append(("SUCCESS" if ok else "FAIL") if ok is not None else "")
        for k, fmt in (("impact_v_vertical", "v_z={:.2f}"), ("impact_v_horizontal", "v_h={:.2f}"),
                       ("tilt_at_impact_deg", "tilt={:.1f}°"), ("first_contact", "first={}")):
            v = info.get(k)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                parts.append(fmt.format(v))
    if parts:
        fig.suptitle("  ".join(p for p in parts if p), fontsize=9)
