"""Figures for the IEEE-style paper (paper/main.tex).

Reads evaluation outputs in runs/ and reports/, writes PNG/PDF figures to paper/figures/.
Run from the project root:  python paper/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "paper" / "figures"
RUNS = ROOT / "runs"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8,
                     "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "figure.dpi": 200})


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in open(p)]


def braking_impulse(H: float, cfg: dict, m: float) -> float:
    """Ideal impulse of the baseline braking profile for a release with feet at height H."""
    b = cfg["baseline"]
    g, vt, a, ht = cfg["sim"]["gravity"], b["target_speed"], b["brake_decel_max"], b["target_height"]
    hb = (2 * g * H - vt ** 2 + 2 * a * ht) / (2 * (g + a))   # profile meets free-fall curve
    v0 = np.sqrt(2 * g * (H - hb))
    return m * (v0 - vt) * (1 + g / a) + m * g * ht / vt          # brake + constant-speed hold


def fig_theory(cfg: dict) -> dict:
    rs = load_jsonl(RUNS / "eval_baseline_t1" / "results.jsonl")
    H = np.array([r["episode_params"]["drop_height"] for r in rs])
    ms = np.array([r["episode_params"]["mass_scale"] for r in rs])
    mass = cfg["human"]["mass"] * ms + sum(v["total"] for v in cfg["suit"]["masses"].values())
    P = np.array([r["impulse_gas_pods"] for r in rs])
    th = np.array([braking_impulse(h, cfg, mm) for h, mm in zip(H, mass)])
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.scatter(th, P, s=3, alpha=0.45, color="#d1561b", lw=0, label="simulated episodes")
    lim = [400, 1900]
    ax.plot(lim, lim, "k--", lw=0.8, label="ideal profile (Eq. 7)")
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_xlabel("first-principles braking impulse [N s]")
    ax.set_ylabel("simulated lift-pod impulse [N s]")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "theory_vs_sim.pdf")
    plt.close(fig)
    ratio = P / th
    stats = {"corr": float(np.corrcoef(th, P)[0, 1]), "ratio_mean": float(ratio.mean()),
             "ratio_p5": float(np.percentile(ratio, 5)), "ratio_p95": float(np.percentile(ratio, 95))}
    bins = []
    for lo in (1, 2, 3, 4):
        s = (H >= lo) & (H < lo + 1)
        bins.append({"bin": f"{lo}-{lo + 1}", "n": int(s.sum()), "H": float(H[s].mean()),
                     "theory": float(th[s].mean()), "pods": float(P[s].mean()),
                     "pods_sd": float(P[s].std())})
    stats["bins"] = bins
    return stats


def filmstrip(video: Path, out: Path, n: int = 6, start: float = 0.0, stop: float = 1.0,
              crop: tuple[float, float] = (0.15, 0.85)) -> None:
    frames = iio.imread(video)
    idx = np.linspace(start * (len(frames) - 1), stop * (len(frames) - 1), n).astype(int)
    w = frames.shape[2]
    c0, c1 = int(crop[0] * w), int(crop[1] * w)
    strip = np.concatenate([frames[i][:, c0:c1] for i in idx], axis=1)
    iio.imwrite(out, strip[::2, ::2])


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "runs" / "eval_baseline_t1" / "config.yaml"))
    stats = fig_theory(cfg)
    json.dump(stats, open(FIG / "theory_stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))
    filmstrip(FIG / "t1_5m_tilt.mp4", FIG / "strip_t1.png", start=0.05, stop=0.95)
    filmstrip(FIG / "t1_nothrust.mp4", FIG / "strip_t1_off.png", n=4, start=0.2, stop=0.95)
    filmstrip(ROOT / "reports/videos/t2_jump/best_0_seed94.mp4", FIG / "strip_t2.png",
              start=0.15, stop=0.98, crop=(0.05, 0.95))
    filmstrip(ROOT / "reports/videos/t3_push/worst_0_seed140.mp4", FIG / "strip_t3.png",
              start=0.1, stop=0.9)
    for name in ("training_curve.png", "t3_stress_by_direction.png", "t1_baseline_vs_residual.png",
                 "t1_baseline_hist_impact.png"):
        (FIG / name).write_bytes((ROOT / "reports/figures" / name).read_bytes())


if __name__ == "__main__":
    main()
