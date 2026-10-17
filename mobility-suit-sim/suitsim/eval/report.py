"""Write evaluation outputs: summary JSON, per-episode CSV/JSONL, markdown, histograms."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from suitsim.interfaces import BUDGET_KEYS, EPISODE_INFO_KEYS  # noqa: E402

from .metrics import primary_failure  # noqa: E402


def _json_default(o: Any):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    return path


def write_jsonl(results: Iterable[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=_json_default) + "\n")
    return path


def flatten_result(r: dict, limits: dict | None = None) -> dict:
    """Flat row for CSV: info keys, episode_params.* columns, failure mode."""
    row = {"seed": r.get("seed")}
    for k in EPISODE_INFO_KEYS:
        row[k] = r.get(k)
    for k, v in (r.get("episode_params") or {}).items():
        row[f"param.{k}"] = v
    row["failure_mode"] = primary_failure(r, limits)
    for k in ("episode_return", "steps", "terminated", "truncated", "wall_time", "error"):
        if k in r:
            row[k] = r[k]
    for k, v in r.items():   # extra scalar keys the env may add
        if k not in row and (v is None or isinstance(v, (int, float, str, bool))):
            row[k] = v
    return row


def write_results_csv(results: Sequence[dict], path: str | Path,
                      limits: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [flatten_result(r, limits) for r in results]
    cols: list[str] = []
    for row in rows:
        cols.extend(c for c in row if c not in cols)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return path


# --------------------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------------------
def _fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (float, np.floating)):
        if not math.isfinite(v):
            return "–"
        return f"{v:.{nd}g}" if abs(v) >= 1e4 or (0 < abs(v) < 1e-3) else f"{v:.{nd}f}"
    return str(v)


def md_table(rows: Sequence[dict] | dict, columns: Sequence[str] | None = None) -> str:
    """Markdown table from a list of dicts (or a dict -> two-column key/value table)."""
    if isinstance(rows, dict):
        rows = [{"key": k, "value": v} for k, v in rows.items()]
    rows = list(rows)
    if not rows:
        return "_(no rows)_\n"
    if columns is None:
        columns = []
        for r in rows:
            columns.extend(c for c in r if c not in columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c)) for c in columns) + " |")
    return "\n".join(lines) + "\n"


def _pct(r: dict) -> str:
    lo, hi = r["ci95"]
    return f"{100 * r['rate']:.1f} % [{100 * lo:.1f}, {100 * hi:.1f}] ({r['k']}/{r['n']})"


def summary_markdown(summary: dict, title: str = "Evaluation summary") -> str:
    s = summary
    out = [f"## {title}\n"]
    if s.get("task"):
        out.append(f"Task: `{s['task']}`, episodes: {s['n']}\n")
    out.append(md_table([
        {"metric": "success", "value (95 % Wilson CI)": _pct(s["success"])},
        {"metric": "safe landing", "value (95 % Wilson CI)": _pct(s["safe"])},
        {"metric": "bad contact", "value (95 % Wilson CI)": _pct(s["bad_contact"])},
        {"metric": "airbags deployed", "value (95 % Wilson CI)": _pct(s["airbags_deployed"])},
        {"metric": "within all budgets",
         "value (95 % Wilson CI)": _pct(s["budget_usage"]["all_within"])},
    ]))
    out.append("\n### Distributions\n")
    out.append(md_table([{"quantity": k, **{c: v[c] for c in ("mean", "std", "p50", "p95", "max")}}
                         for k, v in s["stats"].items()]))
    out.append("\n### Budget usage (T4)\n")
    rows = []
    for bk in BUDGET_KEYS:
        b = s["budget_usage"].get(bk)
        if not b:
            continue
        rows.append({"budget": bk, "capacity": b["budget"], "mean used": b["used"]["mean"],
                     "p95 used": b["used"]["p95"], "max used": b["used"]["max"],
                     "p95 frac": b["used_frac"]["p95"], "within": _pct(b["within"])})
    out.append(md_table(rows))
    out.append("\n### Failure modes (primary label)\n")
    f = s["failures"]
    out.append(md_table([{"mode": m, "count": c, "fraction": f["primary_frac"][m],
                          "any-label count": f["any"][m]}
                         for m, c in f["primary"].items()]))
    out.append("\nFirst contact: " + ", ".join(f"`{k}`: {v}" for k, v in
                                               f["first_contact"].items()) + "\n")
    return "\n".join(out)


def write_markdown_report(sections: Sequence[tuple[str, Any]] | dict, out_path: str | Path,
                          title: str | None = None) -> Path:
    """Generic markdown report writer (used for the final M7 report).

    ``sections``: list of ``(heading, content)`` (or an ordered dict). ``content`` may be
    a str (verbatim markdown), a summary dict from :func:`summarize` (rendered with
    :func:`summary_markdown`), any other dict / list of dicts (tables), a path to a
    ``.png``/``.gif``/``.mp4`` (embedded / linked relative to the report), or a list
    mixing these.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    items = list(sections.items()) if isinstance(sections, dict) else list(sections)

    def render(c: Any) -> str:
        if c is None:
            return ""
        if isinstance(c, Path) or (isinstance(c, str) and os.path.splitext(c)[1].lower()
                                   in (".png", ".jpg", ".gif", ".svg", ".mp4")
                                   and "\n" not in c):
            p = Path(c)
            try:
                rel = os.path.relpath(p.resolve(), out_path.parent.resolve())
            except ValueError:
                rel = str(p)
            if p.suffix.lower() == ".mp4":
                return f"[video: {p.name}]({rel})\n"
            return f"![{p.stem}]({rel})\n"
        if isinstance(c, str):
            return c.rstrip() + "\n"
        if isinstance(c, dict) and {"success", "safe", "stats", "failures"} <= set(c):
            return summary_markdown(c, title="Summary")
        if isinstance(c, dict):
            return md_table(c)
        if isinstance(c, (list, tuple)) and c and all(isinstance(x, dict) for x in c):
            return md_table(c)
        if isinstance(c, (list, tuple)):
            return "\n".join(render(x) for x in c)
        return str(c) + "\n"

    parts = [f"# {title}\n"] if title else []
    for heading, content in items:
        parts.append(f"## {heading}\n")
        parts.append(render(content))
    out_path.write_text("\n".join(parts))
    return out_path


# --------------------------------------------------------------------------------------
# Histograms
# --------------------------------------------------------------------------------------
def _hist(ax, values, label, limit=None, color="C0"):
    v = np.asarray([x for x in values if x is not None], dtype=float)
    v = v[np.isfinite(v)]
    if v.size:
        ax.hist(v, bins=min(40, max(5, int(np.sqrt(v.size)) + 1)), color=color, alpha=0.8,
                edgecolor="white", linewidth=0.5)
        ax.axvline(np.median(v), color="k", lw=1, ls="--", label=f"median {np.median(v):.3g}")
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    if limit is not None:
        ax.axvline(limit, color="C3", lw=1.5, label=f"limit {limit:g}")
    ax.set_xlabel(label)
    ax.set_ylabel("episodes")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)


def plot_histograms(results: Sequence[dict], out_dir: str | Path, summary: dict | None = None,
                    prefix: str = "") -> list[Path]:
    """Write hist_impact.png, hist_tilt.png, hist_budgets.png, failure_modes.png."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lim = (summary or {}).get("limits", {})
    bud = (summary or {}).get("budgets", {})
    landed = [r for r in results if str(r.get("first_contact", "none")) != "none"]
    paths = []

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    _hist(axes[0], [abs(r["impact_v_vertical"]) for r in landed if "impact_v_vertical" in r],
          "vertical impact speed [m/s]", lim.get("max_vertical_speed"))
    _hist(axes[1], [abs(r["impact_v_horizontal"]) for r in landed if "impact_v_horizontal" in r],
          "horizontal impact speed [m/s]", lim.get("max_horizontal_speed"), color="C1")
    fig.suptitle(f"Impact speed at first contact ({len(landed)} landed episodes)")
    fig.tight_layout()
    paths.append(out_dir / f"{prefix}hist_impact.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.6))
    _hist(ax, [r.get("tilt_at_impact_deg") for r in landed], "tilt at impact [deg]",
          lim.get("max_tilt_deg"), color="C2")
    fig.tight_layout()
    paths.append(out_dir / f"{prefix}hist_tilt.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(BUDGET_KEYS), figsize=(4.2 * len(BUDGET_KEYS), 3.6))
    units = {"impulse_gas_pods": "N·s", "impulse_gas_jets": "N·s", "battery_wh": "Wh"}
    for ax, bk, c in zip(np.atleast_1d(axes), BUDGET_KEYS, ("C0", "C4", "C5")):
        _hist(ax, [r.get(bk) for r in results], f"{bk} used [{units[bk]}]", bud.get(bk), c)
    fig.suptitle("Budget used per episode, by thruster class (pods / micro-jets / fans)")
    fig.tight_layout()
    paths.append(out_dir / f"{prefix}hist_budgets.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)

    modes: dict[str, int] = {}
    for r in results:
        m = primary_failure(r, lim or None)
        modes[m] = modes.get(m, 0) + 1
    fig, ax = plt.subplots(figsize=(6, 3.6))
    names = sorted(modes, key=lambda k: (k != "none", -modes[k]))
    ax.barh(names, [modes[n] for n in names],
            color=["C2" if n == "none" else "C3" for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("episodes")
    ax.set_title("Outcome ('none' = success)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    paths.append(out_dir / f"{prefix}failure_modes.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)
    return paths


def write_eval_outputs(results: Sequence[dict], summary: dict, out_dir: str | Path,
                       title: str = "Evaluation") -> dict[str, Path]:
    """summary.json, results.csv, results.jsonl, summary.md and histogram PNGs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lim = summary.get("limits")
    files = {
        "summary_json": write_json(summary, out_dir / "summary.json"),
        "results_csv": write_results_csv(results, out_dir / "results.csv", lim),
        "results_jsonl": write_jsonl(results, out_dir / "results.jsonl"),
    }
    plots = plot_histograms(results, out_dir / "plots", summary)
    md = summary_markdown(summary, title=title)
    md += "\n### Plots\n\n" + "\n".join(f"![{p.stem}](plots/{p.name})" for p in plots) + "\n"
    (out_dir / "summary.md").write_text(md)
    files["summary_md"] = out_dir / "summary.md"
    for p in plots:
        files[p.stem] = p
    return files
