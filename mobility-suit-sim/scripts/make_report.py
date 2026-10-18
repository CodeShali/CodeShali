#!/usr/bin/env python
"""Build the M7 report (reports/M7_report.md + figures + videos) from the evaluation runs.

    python scripts/make_report.py            # reads runs/*, writes reports/ and results/

Inputs are the evaluation folders written by ``scripts/evaluate.py`` (see RUNS below) and
the PPO training folders (eval.csv). ``reports/`` is self-contained (relative links); the
summary JSONs are also copied to ``results/m7_<name>_summary.json``.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from suitsim.eval.metrics import primary_failure, rate, success_by_param  # noqa: E402
from suitsim.eval.report import md_table, write_json, write_markdown_report  # noqa: E402

RUNS = ROOT / "runs"
REPORT = ROOT / "reports"
FIG = REPORT / "figures"
VID = REPORT / "videos"
RESULTS = ROOT / "results"

# name -> (run dir, label)
EVALS = {
    "t1_baseline": ("eval_baseline_t1", "T1 drop — baseline"),
    "t1_residual": ("eval_residual_t1", "T1 drop — residual PPO"),
    "t2_jump": ("eval_t2", "T2 boosted jump — baseline"),
    "t3_push": ("eval_t3_final", "T3 push 200 N — baseline"),
    "t3_push_side_jets": ("eval_t3_side_jets", "T3 push 200 N — side-jet suit"),
    "t3_400N_default": ("stress_t3_400N_default", "T3 stress 400 N — default suit"),
    "t3_400N_side_jets": ("stress_t3_400N_side_jets", "T3 stress 400 N — side-jet suit"),
    "t3_500N_default": ("stress_t3_500N_default", "T3 stress 500 N — default suit"),
    "t3_500N_side_jets": ("stress_t3_500N_side_jets", "T3 stress 500 N — side-jet suit"),
    "t1_side_jets": ("eval_t1_side_jets", "T1 drop — side-jet suit (200 eps)"),
    "t1_pods_fwd": ("eval_t1_pods_fwd", "T1 drop — pods 3.6 cm forward (200 eps)"),
}
# evaluations whose 5 worst + 5 best videos are copied into reports/videos/<name>/
VIDEO_EVALS = ["t1_baseline", "t1_residual", "t2_jump", "t3_push", "t3_push_side_jets",
               "t3_500N_default", "t3_500N_side_jets"]

# Acceptance targets (project spec). T2's >= 90 % is also quoted in configs/task_jump.yaml.
TARGETS = {"T1": 0.95, "T2": 0.90, "T3": 0.90}


# --------------------------------------------------------------------------------------
def load(name: str) -> tuple[dict, list[dict]] | None:
    d = RUNS / EVALS[name][0]
    if not (d / "summary.json").exists():
        return None
    s = json.loads((d / "summary.json").read_text())
    res = [json.loads(line) for line in open(d / "results.jsonl")]
    return s, res


def pct(r: dict) -> str:
    lo, hi = r["ci95"]
    return f"{100 * r['rate']:.1f} % [{100 * lo:.1f}, {100 * hi:.1f}] ({r['k']}/{r['n']})"


def mean_ci(x) -> str:
    x = np.asarray(x, float)
    se = x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float("nan")
    return f"{x.mean():.1f} ± {1.96 * se:.1f}"


def paired(a: list[dict], b: list[dict], key) -> dict:
    """Paired difference b - a over common seeds (mean, 95 % CI half-width, n)."""
    ma = {r["seed"]: r for r in a if "error" not in r}
    mb = {r["seed"]: r for r in b if "error" not in r}
    seeds = sorted(set(ma) & set(mb))
    d = np.array([key(mb[s]) - key(ma[s]) for s in seeds], float)
    xa = np.array([key(ma[s]) for s in seeds], float)
    return {"n": len(seeds), "mean_a": xa.mean(), "mean_b": xa.mean() + d.mean(),
            "diff": d.mean(), "ci": 1.96 * d.std(ddof=1) / math.sqrt(len(d)),
            "rel": d.mean() / xa.mean()}


def total_impulse(r):
    return float(r.get("impulse_gas_pods", 0.0)) + float(r.get("impulse_gas_jets", 0.0))


def subset(res: list[dict], seeds) -> list[dict]:
    ss = set(seeds)
    return [r for r in res if r["seed"] in ss]


def stat(s: dict, k: str, f: str = "mean") -> float:
    return s["stats"][k][f]


def bud(s: dict, k: str, f: str = "mean") -> float:
    return s["budget_usage"][k]["used"][f]


def bins_table(res: list[dict], param: str, edges, fmt="{:.2g}") -> list[dict]:
    rows = []
    for b in success_by_param(res, param, edges):
        rows.append({param: f"{fmt.format(b['lo'])}–{fmt.format(b['hi'])}", "n": b["n"],
                     "success": f"{100 * b['rate']:.1f} %",
                     "95 % CI": f"[{100 * b['ci95'][0]:.1f}, {100 * b['ci95'][1]:.1f}]"})
    return rows


# --------------------------------------------------------------------------------------
def copy_figures(data: dict) -> dict[str, list[Path]]:
    FIG.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in ("t1_baseline", "t1_residual", "t2_jump", "t3_push", "t3_push_side_jets"):
        if name not in data:
            continue
        src = RUNS / EVALS[name][0] / "plots"
        out[name] = []
        for p in sorted(src.glob("*.png")):
            dst = FIG / f"{name}_{p.name}"
            shutil.copy2(p, dst)
            out[name].append(dst)
    return out


def copy_videos(data: dict) -> dict[str, list[dict]]:
    out = {}
    for name in VIDEO_EVALS:
        if name not in data:
            continue
        s, _ = data[name]
        src = RUNS / EVALS[name][0] / "videos"
        dst = VID / name
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        rows = []
        for v in s.get("videos", []):
            shutil.copy2(src / v["video"], dst / v["video"])
            if v["label"] == "worst" and v["rank"] == 0 and (src / v["plot"]).exists():
                shutil.copy2(src / v["plot"], dst / v["plot"])
            rows.append({**v, "path": dst / v["video"]})
        out[name] = rows
    return out


def video_table(rows: list[dict], res: list[dict]) -> str:
    by_seed = {r["seed"]: r for r in res}
    lines = ["| clip | seed | outcome | details |", "|---|---|---|---|"]
    for v in rows:
        r = by_seed.get(v["seed"], {})
        rel = v["path"].relative_to(REPORT)
        p = r.get("episode_params", {})
        if "drop_height" in p:
            det = (f"h {p['drop_height']:.2f} m, tilt {p['tilt_deg']:.0f}°, |ω| {p['angvel']:.2f} "
                   f"rad/s; vz {abs(r.get('impact_v_vertical', 0)):.2f}, vh "
                   f"{abs(r.get('impact_v_horizontal', 0)):.2f} m/s, tilt@impact "
                   f"{r.get('tilt_at_impact_deg', 0):.1f}°")
        elif "push_direction_deg" in p:
            det = (f"{p['push_force']:.0f} N at {p['push_direction_deg']:.0f}° "
                   f"({p['push_lateral_deg']:.0f}° off sagittal); final tilt "
                   f"{r.get('final_tilt_deg', 0):.1f}°, jets {r.get('impulse_gas_jets', 0):.0f} N·s")
        else:
            det = (f"vz {abs(r.get('impact_v_vertical', 0)):.2f}, vh "
                   f"{abs(r.get('impact_v_horizontal', 0)):.2f} m/s; pods "
                   f"{r.get('impulse_gas_pods', 0):.0f}, jets {r.get('impulse_gas_jets', 0):.0f} N·s")
        out = "success" if v.get("success") else f"**fail: {v.get('failure_mode')}**"
        lines.append(f"| [{v['label']} #{v['rank']}]({rel.as_posix()}) | {v['seed']} | {out} | {det} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
def plot_training_curve(path: Path) -> Path | None:
    import csv

    def read(run):
        f = RUNS / run / "eval.csv"
        if not f.exists():
            return None
        rows = list(csv.DictReader(open(f)))
        return (np.array([float(r["timesteps"]) for r in rows]),
                np.array([float(r["impulse_total"]) for r in rows]),
                np.array([float(r["success_rate"]) for r in rows]),
                np.array([float(r["impulse_pods"]) for r in rows]),
                np.array([float(r["impulse_jets"]) for r in rows]))

    b, c, q = read("ppo_t1_b"), read("ppo_t1_c"), read("ppo_quick_a")
    if b is None or c is None:
        return None
    base = json.loads((RUNS / "ppo_t1_c" / "eval_baseline.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    ax.plot(b[0] / 1e3, b[1], "o-", color="C0", label="ppo_t1_b (log std −1.2)")
    ax.plot(c[0] / 1e3, c[1], "o-", color="C1", label="ppo_t1_c (init from b @ 400k)")
    if q is not None:
        ax.plot(q[0] / 1e3, q[1], "s--", color="C7", ms=4, label="ppo_quick_a (log std −2.0)")
    ax.axhline(base["impulse_total"], color="k", ls=":", label=f"baseline {base['impulse_total']:.0f}")
    ax.set_xlabel("PPO timesteps [k]")
    ax.set_ylabel("total gas impulse (pods + jets) [N·s]")
    ax.set_title("Mean impulse, 50 held-out eval episodes")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    ax.plot(b[0] / 1e3, 100 * b[2], "o-", color="C0", label="ppo_t1_b")
    ax.plot(c[0] / 1e3, 100 * c[2], "o-", color="C1", label="ppo_t1_c")
    if q is not None:
        ax.plot(q[0] / 1e3, 100 * q[2], "s--", color="C7", ms=4, label="ppo_quick_a")
    ax.axhline(100 * base["success_rate"], color="k", ls=":", label="baseline")
    ax.set_ylim(90, 101)
    ax.set_xlabel("PPO timesteps [k]")
    ax.set_ylabel("success (safe landing) [%]")
    ax.set_title("Success rate (n = 50, 1 failure = 2 %)")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_compare_impulse(a: list[dict], b: list[dict], path: Path, la: str, lb: str) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (key, lab) in zip(axes, [("impulse_gas_pods", "pod gas [N·s]"),
                                     ("impulse_gas_jets", "jet gas [N·s]"),
                                     ("impact_v_vertical", "|vertical impact speed| [m/s]")]):
        va = np.abs([r[key] for r in a]); vb = np.abs([r[key] for r in b])
        bins = np.linspace(min(va.min(), vb.min()), max(va.max(), vb.max()), 40)
        ax.hist(va, bins=bins, alpha=0.6, label=f"{la} (mean {va.mean():.3g})", color="C0")
        ax.hist(vb, bins=bins, alpha=0.6, label=f"{lb} (mean {vb.mean():.3g})", color="C1")
        ax.set_xlabel(lab); ax.set_ylabel("episodes")
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("T1, 1000 paired episodes (seeds 0–999)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_push_stress(data: dict, path: Path) -> Path | None:
    edges = [0, 15, 30, 45, 60, 75, 90.001]
    series = [(k, EVALS[k][1]) for k in ("t3_400N_default", "t3_400N_side_jets",
                                          "t3_500N_default", "t3_500N_side_jets") if k in data]
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.8))
    centers = 0.5 * (np.array(edges[:-1]) + np.array(edges[1:]))
    styles = {"t3_400N_default": ("C0", "--o"), "t3_400N_side_jets": ("C1", "--s"),
              "t3_500N_default": ("C0", "-o"), "t3_500N_side_jets": ("C1", "-s")}
    for k, lab in series:
        rows = success_by_param(data[k][1], "push_lateral_deg", edges)
        x = [0.5 * (r["lo"] + r["hi"]) for r in rows]
        y = [100 * r["rate"] for r in rows]
        c, st = styles[k]
        ax.plot(x, y, st, color=c, label=lab.replace("T3 stress ", ""))
    ax.set_xticks(centers, [f"{a:.0f}–{b:.0f}" for a, b in zip(edges[:-1], edges[1:])])
    ax.set_xlabel("push direction, degrees off the sagittal axis (90 = pure lateral)")
    ax.set_ylabel("stayed on beam [%]")
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("T3 stress: success vs push direction (100 eps per curve)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
def main() -> int:
    data = {k: v for k in EVALS if (v := load(k)) is not None}
    missing = [k for k in EVALS if k not in data]
    if missing:
        print("missing runs:", missing)
    REPORT.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    figs = copy_figures(data)
    vids = copy_videos(data)
    RESULTS.mkdir(exist_ok=True)
    for k, (s, _) in data.items():
        write_json(s, RESULTS / f"m7_{k}_summary.json")

    S = {k: v[0] for k, v in data.items()}
    R = {k: v[1] for k, v in data.items()}
    sec: list[tuple[str, object]] = []

    # ---------------------------------------------------------------- executive summary
    acc = []

    def acc_row(task, name, metric, target):
        s = S[name]
        r = s["success"]
        ok = r["rate"] >= target
        acc.append({"task": task, "evaluation": EVALS[name][1], "episodes": s["n"],
                    "metric": metric, "result (95 % Wilson CI)": pct(r),
                    "target": f"≥ {100 * target:.0f} %",
                    "pass": ("**PASS**" if ok else "FAIL")
                    + ("" if r["ci95"][0] >= target else " (CI lower bound below target)")})
    acc_row("T1", "t1_baseline", "safe landing", TARGETS["T1"])
    acc_row("T1", "t1_residual", "safe landing", TARGETS["T1"])
    acc_row("T2", "t2_jump", "on platform, safe", TARGETS["T2"])
    acc_row("T3", "t3_push", "stayed on beam", TARGETS["T3"])
    for name in ("t1_baseline", "t1_residual", "t2_jump", "t3_push"):
        s = S[name]
        w = s["budget_usage"]["all_within"]
        acc.append({"task": "T4", "evaluation": EVALS[name][1], "episodes": s["n"],
                    "metric": "within all budgets", "result (95 % Wilson CI)": pct(w),
                    "target": "100 % of episodes",
                    "pass": "**PASS**" if w["rate"] >= 1.0 else "FAIL"})

    pr = paired(R["t1_baseline"], R["t1_residual"], total_impulse)
    pp = paired(R["t1_baseline"], R["t1_residual"], lambda r: r["impulse_gas_pods"])
    pj = paired(R["t1_baseline"], R["t1_residual"], lambda r: r["impulse_gas_jets"])
    pv = paired(R["t1_baseline"], R["t1_residual"], lambda r: abs(r["impact_v_vertical"]))

    exec_md = [
        "Simulation of a thruster-based wearable mobility suit (91 kg human + 9 kg suit, "
        "2 lift pods, 2 fans, 40 micro-jets, 3 airbags) in MuJoCo, with a sensor-only state "
        "estimator, a constrained thrust allocator, model-based controllers for the three "
        "tasks and a residual PPO policy for T1. Every number below comes from fully "
        "randomised episodes (human mass ±10 %, thrust ±15 %, thruster lag ±50 %, sensor "
        "noise ×0.5–2, ground friction 0.5–1.0) evaluated with fixed seeds; rates carry "
        "Wilson 95 % intervals.\n",
        md_table(acc),
        "\n**Headlines**\n",
        f"* **T1 drop landing (1–5 m, ≤30° tilt, ≤2 rad/s spin)**: baseline "
        f"{pct(S['t1_baseline']['success'])}; all failures are horizontal-speed "
        f"(> 1 m/s) landings from low drops with high tilt + spin — no vertical, tilt or "
        f"bad-contact failures, airbags never needed.",
        f"* **Residual PPO** (850k steps, 4 % of the 20 M-step spec budget): "
        f"{pct(S['t1_residual']['success'])} with **{100 * pr['rel']:+.1f} % total gas** "
        f"(paired {pr['diff']:+.1f} ± {pr['ci']:.1f} N·s), pods {100 * pp['rel']:+.1f} %, "
        f"jets {100 * pj['rel']:+.1f} %. It saves gas by landing faster "
        f"(|vz| {pv['mean_a']:.2f} → {pv['mean_b']:.2f} m/s, limit 2.0).",
        f"* **T2 boosted jump onto 2.5 m platform**: {pct(S['t2_jump']['success'])}; "
        f"the binding budget is jet gas (mean {bud(S['t2_jump'], 'impulse_gas_jets'):.0f} of 600 N·s).",
        f"* **T3 push recovery on 0.3 m beam (200 N × 0.2 s)**: {pct(S['t3_push']['success'])}; "
        f"jets {bud(S['t3_push'], 'impulse_gas_jets'):.0f} N·s mean, pods ≈ 0.",
    ]
    if "t3_500N_default" in S and "t3_500N_side_jets" in S:
        exec_md.append(
            f"* **Thruster layout**: the default suit is weak laterally. At 500 N pushes the "
            f"default suit keeps {pct(S['t3_500N_default']['success'])} on the beam, the "
            f"+4 side-jet variant {pct(S['t3_500N_side_jets']['success'])}.")
    if "t1_pods_fwd" in S:
        pj2 = paired(subset(R["t1_baseline"], range(200)), R["t1_pods_fwd"],
                     lambda r: r["impulse_gas_jets"])
        exec_md.append(
            f"* **Pod placement**: moving the pods 3.6 cm forward (thrust line through the "
            f"CoM) changes T1 jet gas by {100 * pj2['rel']:+.1f} % "
            f"({pj2['diff']:+.1f} ± {pj2['ci']:.1f} N·s paired, 200 eps).")
    sec.append(("1. Executive summary", "\n".join(exec_md)))

    # ---------------------------------------------------------------- system description
    sec.append(("2. System description", SYSTEM_MD))

    # ---------------------------------------------------------------- T1
    t1 = []
    for name in ("t1_baseline", "t1_residual"):
        s, res = S[name], R[name]
        t1.append(f"### {EVALS[name][1]} ({s['n']} episodes, seeds {s['seeds'][0]}–{s['seeds'][1]})\n")
        t1.append(md_table([
            {"metric": "safe landing", "value": pct(s["success"])},
            {"metric": "bad contact (head/torso/pelvis first)", "value": pct(s["bad_contact"])},
            {"metric": "airbags deployed", "value": pct(s["airbags_deployed"])},
            {"metric": "|vz| at impact mean / p95 / max [m/s]",
             "value": f"{abs(stat(s, 'impact_v_vertical_landed')):.2f} / "
                      f"{s['stats']['impact_v_vertical_landed']['p95']:.2f} / "
                      f"{s['stats']['impact_v_vertical_landed']['max']:.2f}"},
            {"metric": "|vh| at impact mean / p95 / max [m/s]",
             "value": f"{stat(s, 'impact_v_horizontal_landed'):.2f} / "
                      f"{s['stats']['impact_v_horizontal_landed']['p95']:.2f} / "
                      f"{s['stats']['impact_v_horizontal_landed']['max']:.2f}"},
            {"metric": "tilt at impact mean / p95 / max [deg]",
             "value": f"{stat(s, 'tilt_at_impact_deg_landed'):.1f} / "
                      f"{s['stats']['tilt_at_impact_deg_landed']['p95']:.1f} / "
                      f"{s['stats']['tilt_at_impact_deg_landed']['max']:.1f}"},
        ]))
        fails = [r for r in res if not r.get("success")]
        t1.append(f"\n**Failures ({len(fails)})** — primary label and episode parameters:\n")
        rows = []
        for r in fails:
            p = r["episode_params"]
            rows.append({"seed": r["seed"], "mode": primary_failure(r, s["limits"]),
                         "drop h [m]": p["drop_height"], "tilt [deg]": p["tilt_deg"],
                         "|ω| [rad/s]": p["angvel"], "|vz| [m/s]": abs(r["impact_v_vertical"]),
                         "|vh| [m/s]": abs(r["impact_v_horizontal"]),
                         "tilt@impact [deg]": r["tilt_at_impact_deg"],
                         "mass ×": p["mass_scale"]})
        t1.append(md_table(rows))
        t1.append("\n**Success by drop height / initial tilt / spin rate**\n")
        t1.append(md_table(bins_table(res, "drop_height", [1, 1.5, 2, 3, 4, 5.001])))
        t1.append("")
        t1.append(md_table(bins_table(res, "tilt_deg", [0, 10, 20, 25, 30.001], "{:.0f}")))
        t1.append("")
        t1.append(md_table(bins_table(res, "angvel", [0, 1, 1.5, 2, 2.5, 3.5])))
        t1.append("")
        for p in figs.get(name, []):
            t1.append(f"![{p.stem}](figures/{p.name})\n")
    t1.append(T1_FAILURE_MD)
    sec.append(("3. T1 — drop landing", "\n".join(t1)))

    # ---------------------------------------------------------------- baseline vs residual
    cmpf = plot_compare_impulse(R["t1_baseline"], R["t1_residual"],
                                FIG / "t1_baseline_vs_residual.png", "baseline", "residual PPO")
    cmp_rows = []
    for lab, key in [("total gas [N·s]", total_impulse),
                     ("pod gas [N·s]", lambda r: r["impulse_gas_pods"]),
                     ("jet gas [N·s]", lambda r: r["impulse_gas_jets"]),
                     ("fan battery [Wh]", lambda r: r["battery_wh"]),
                     ("|vz| at impact [m/s]", lambda r: abs(r["impact_v_vertical"])),
                     ("|vh| at impact [m/s]", lambda r: abs(r["impact_v_horizontal"])),
                     ("tilt at impact [deg]", lambda r: r["tilt_at_impact_deg"])]:
        q = paired(R["t1_baseline"], R["t1_residual"], key)
        cmp_rows.append({"quantity": lab, "baseline": f"{q['mean_a']:.3g}",
                         "residual": f"{q['mean_b']:.3g}",
                         "paired diff (95 % CI)": f"{q['diff']:+.3g} ± {q['ci']:.2g}",
                         "relative": f"{100 * q['rel']:+.1f} %"})
    cmp_rows.insert(0, {"quantity": "safe landing", "baseline": pct(S["t1_baseline"]["success"]),
                        "residual": pct(S["t1_residual"]["success"]),
                        "paired diff (95 % CI)": "", "relative": ""})
    tc = plot_training_curve(FIG / "training_curve.png")
    cmp_md = [
        "Same 1000 seeds (identical initial states and randomisation draws), so the "
        "differences are paired.\n", md_table(cmp_rows), f"\n![compare](figures/{cmpf.name})\n",
        RL_MD,
    ]
    if tc:
        cmp_md.append(f"![training curve](figures/{tc.name})\n")
    sec.append(("4. Baseline vs residual PPO (T1)", "\n".join(cmp_md)))

    # ---------------------------------------------------------------- T2
    s = S["t2_jump"]
    t2 = [md_table([
        {"metric": "success (foot on top, safe, stayed)", "value": pct(s["success"])},
        {"metric": "pods mean / p95 / max [N·s] (cap 2500)",
         "value": f"{bud(s, 'impulse_gas_pods'):.0f} / {bud(s, 'impulse_gas_pods', 'p95'):.0f} / "
                  f"{bud(s, 'impulse_gas_pods', 'max'):.0f}"},
        {"metric": "jets mean / p95 / max [N·s] (cap 600)",
         "value": f"{bud(s, 'impulse_gas_jets'):.0f} / {bud(s, 'impulse_gas_jets', 'p95'):.0f} / "
                  f"{bud(s, 'impulse_gas_jets', 'max'):.0f}"},
        {"metric": "|vz| at platform contact mean / max [m/s]",
         "value": f"{abs(stat(s, 'impact_v_vertical_landed')):.2f} / "
                  f"{s['stats']['impact_v_vertical_landed']['max']:.2f}"},
        {"metric": "|vh| at platform contact mean / max [m/s]",
         "value": f"{stat(s, 'impact_v_horizontal_landed'):.2f} / "
                  f"{s['stats']['impact_v_horizontal_landed']['max']:.2f}"},
    ]), "", md_table(bins_table(R["t2_jump"], "platform_distance", [1.3, 1.4, 1.5, 1.6, 1.7001]))]
    ex = [r for r in R["t2_jump"] if r["impulse_gas_jets"] >= 599.0]
    t2.append(f"\nJet gas is the T2 bottleneck: {len(ex)} episode(s) exhausted the 600 N·s jet "
              "tank (and still succeeded — the pods carry attitude through the allocator once "
              "the jets are empty). Pods use under half their tank.\n")
    for p in figs.get("t2_jump", []):
        t2.append(f"![{p.stem}](figures/{p.name})\n")
    sec.append(("5. T2 — boosted jump onto a 2.5 m platform", "\n".join(t2)))

    # ---------------------------------------------------------------- T3
    t3 = [T3_INTRO_MD]
    rows = []
    for k in ("t3_push", "t3_push_side_jets", "t3_400N_default", "t3_400N_side_jets",
              "t3_500N_default", "t3_500N_side_jets"):
        if k not in S:
            continue
        s = S[k]
        rows.append({"evaluation": EVALS[k][1], "episodes": s["n"], "stayed on beam": pct(s["success"]),
                     "jets mean [N·s]": f"{bud(s, 'impulse_gas_jets'):.0f}",
                     "jets p95 / max": f"{bud(s, 'impulse_gas_jets', 'p95'):.0f} / "
                                       f"{bud(s, 'impulse_gas_jets', 'max'):.0f}",
                     "pods mean [N·s]": f"{bud(s, 'impulse_gas_pods'):.0f}",
                     "within budgets": f"{100 * s['budget_usage']['all_within']['rate']:.0f} %"})
    t3.append(md_table(rows))
    if "t3_push_side_jets" in R:
        q = paired(R["t3_push"], R["t3_push_side_jets"], lambda r: r["impulse_gas_jets"])
        t3.append(f"\nAt the nominal 200 N the side jets change jet gas by {100 * q['rel']:+.1f} % "
                  f"({q['diff']:+.1f} ± {q['ci']:.1f} N·s paired over {q['n']} seeds).\n")
    t3.append("\n**Success by push direction** (degrees off the sagittal axis; 90° = pure "
              "sideways push, across the beam):\n")
    edges = [0, 15, 30, 45, 60, 75, 90.001]
    for k in ("t3_push", "t3_500N_default", "t3_500N_side_jets"):
        if k in R:
            t3.append(f"*{EVALS[k][1]}*\n")
            t3.append(md_table(bins_table(R[k], "push_lateral_deg", edges, "{:.0f}")))
            t3.append("")
    ps = plot_push_stress(data, FIG / "t3_stress_by_direction.png")
    if ps:
        t3.append(f"![stress](figures/{ps.name})\n")
    for k in ("t3_500N_default",):
        if k in R:
            fails = [r for r in R[k] if not r.get("success")]
            js = [r["impulse_gas_jets"] for r in fails]
            if fails:
                t3.append(f"500 N default-suit failures: {len(fails)}, push direction "
                          f"{np.mean([r['episode_params']['push_lateral_deg'] for r in fails]):.0f}° "
                          f"off sagittal on average; jet gas used {np.mean(js):.0f} N·s mean "
                          f"(max {np.max(js):.0f}).\n")
    for p in figs.get("t3_push", []):
        t3.append(f"![{p.stem}](figures/{p.name})\n")
    sec.append(("6. T3 — push recovery on a 0.3 m beam", "\n".join(t3)))

    # ---------------------------------------------------------------- T4
    t4 = ["Budgets: pod gas 2500 N·s (both pods share one tank), jet gas 600 N·s (all "
          "micro-jets), fan battery 150 Wh. Usage per episode (mean / p95 / max) and the "
          "fraction of episodes that stayed within every budget:\n"]
    rows = []
    for k in ("t1_baseline", "t1_residual", "t2_jump", "t3_push", "t3_push_side_jets",
              "t3_500N_default", "t3_500N_side_jets", "t1_side_jets", "t1_pods_fwd"):
        if k not in S:
            continue
        s = S[k]
        row = {"evaluation": EVALS[k][1], "n": s["n"]}
        for bk, lab in (("impulse_gas_pods", "pods [N·s]"), ("impulse_gas_jets", "jets [N·s]"),
                        ("battery_wh", "fans [Wh]")):
            u = s["budget_usage"][bk]["used"]
            nd = 2 if bk == "battery_wh" else 0
            row[lab] = f"{u['mean']:.{nd}f} / {u['p95']:.{nd}f} / {u['max']:.{nd}f}"
        row["max jets frac"] = f"{100 * s['budget_usage']['impulse_gas_jets']['used_frac']['max']:.0f} %"
        row["within all"] = pct(s["budget_usage"]["all_within"])
        rows.append(row)
    t4.append(md_table(rows))
    t4.append(T4_MD)
    for name in ("t1_baseline", "t2_jump", "t3_push"):
        for p in figs.get(name, []):
            if "budgets" in p.name:
                t4.append(f"*{EVALS[name][1]}*\n\n![{p.stem}](figures/{p.name})\n")
    sec.append(("7. T4 — budgets", "\n".join(t4)))

    # ---------------------------------------------------------------- videos
    vm = [VIDEO_MD]
    for name, rows in vids.items():
        vm.append(f"### {EVALS[name][1]}\n")
        vm.append(video_table(rows, R[name]))
        w0 = [v for v in rows if v["label"] == "worst" and v["rank"] == 0]
        if w0 and (VID / name / w0[0]["plot"]).exists():
            vm.append(f"![worst #0 time series](videos/{name}/{w0[0]['plot']})\n")
    sec.append(("8. Failure-case and best-case videos", "\n".join(vm)))

    # ---------------------------------------------------------------- recommendations
    rec = [REC_INTRO_MD]
    rows = []
    for k_def, k_var, lab in (("t3_push", "t3_push_side_jets", "T3 200 N"),
                              ("t3_400N_default", "t3_400N_side_jets", "T3 400 N"),
                              ("t3_500N_default", "t3_500N_side_jets", "T3 500 N")):
        if k_def in S and k_var in S:
            q = paired(R[k_def], R[k_var], lambda r: r["impulse_gas_jets"])
            rows.append({"case": lab, "default suit": pct(S[k_def]["success"]),
                         "+4 side jets": pct(S[k_var]["success"]),
                         "jet gas default → variant [N·s]": f"{q['mean_a']:.0f} → {q['mean_b']:.0f}"})
    if "t1_side_jets" in S:
        base200 = subset(R["t1_baseline"], range(200))
        q = paired(base200, R["t1_side_jets"], lambda r: r["impulse_gas_jets"])
        rows.append({"case": "T1 drop (200 eps)",
                     "default suit": pct(rate([bool(r["success"]) for r in base200])),
                     "+4 side jets": pct(S["t1_side_jets"]["success"]),
                     "jet gas default → variant [N·s]": f"{q['mean_a']:.0f} → {q['mean_b']:.0f}"})
    rec.append("### 9.1 Add lateral micro-jets (configs/suit_variant_side_jets.yaml)\n")
    rec.append(md_table(rows))
    rec.append(SIDE_JET_MD)
    if "t1_pods_fwd" in S:
        base200 = subset(R["t1_baseline"], range(200))
        rows = []
        for lab, key in [("safe landing", None),
                         ("pod gas [N·s]", lambda r: r["impulse_gas_pods"]),
                         ("jet gas [N·s]", lambda r: r["impulse_gas_jets"]),
                         ("total gas [N·s]", total_impulse),
                         ("|vh| at impact [m/s]", lambda r: abs(r["impact_v_horizontal"])),
                         ("tilt at impact [deg]", lambda r: r["tilt_at_impact_deg"])]:
            if key is None:
                rows.append({"quantity": lab,
                             "default (seeds 0–199)": pct(rate([bool(r["success"]) for r in base200])),
                             "pods +3.6 cm fwd": pct(S["t1_pods_fwd"]["success"]),
                             "paired diff (95 % CI)": "", "relative": ""})
                continue
            q = paired(base200, R["t1_pods_fwd"], key)
            rows.append({"quantity": lab, "default (seeds 0–199)": f"{q['mean_a']:.3g}",
                         "pods +3.6 cm fwd": f"{q['mean_b']:.3g}",
                         "paired diff (95 % CI)": f"{q['diff']:+.3g} ± {q['ci']:.2g}",
                         "relative": f"{100 * q['rel']:+.1f} %"})
        rec.append("### 9.2 Move the lift pods forward onto the CoM (configs/suit_variant_pods_fwd.yaml)\n")
        rec.append(md_table(rows))
        rec.append(PODS_MD)
    rec.append(REC_SUMMARY_MD)
    sec.append(("9. Recommended thruster count / placement changes", "\n".join(rec)))

    sec.append(("10. Limitations and next steps", LIMITS_MD))
    sec.append(("11. Reproducing this report", REPRO_MD))

    out = write_markdown_report(sec, REPORT / "M7_report.md",
                                title="Mobility-suit simulator — M7 evaluation report")
    print("wrote", out)
    return 0


# --------------------------------------------------------------------------------------
SYSTEM_MD = """
**Plant (suitsim/model, MuJoCo ≥ 3).** A procedurally generated MJCF (`suitsim/model/builder.py`,
every physical number from `configs/suit_default.yaml`): 15-segment human (91 kg, 1.80 m,
de Leva mass fractions, 22 hinge joints with PD posture servos, torque limits and passive
leg damping that absorbs landing energy) plus a 9 kg suit (hip pods/tanks, tube frame,
airbags, compute, shell). 1 ms physics step, whole-body quadratic drag. Three airbags (collar,
chest, hips) are stowed/deployed *by geometry* (30 ms inflation) and have soft contact
parameters. Domain randomisation (`suitsim/model/randomize.py`): human segment mass ±10 %,
per-thruster f_max ±15 %, thruster time constants ±50 %, sensor noise ×0.5–2, ground friction
0.5–1.0.

**Actuators (suitsim/actuators).** 44 thrusters, push-only, first-order lag with shared
consumable budgets:

| class | count | f_max | τ | budget |
|---|---|---|---|---|
| lift pod (hips, thrust +z) | 2 | 1500 N | 5 ms | pod gas 2500 N·s (shared) |
| ducted fan (back of hips) | 2 | 300 N | 100 ms | battery 150 Wh, P = k f^1.5, 1.2 kW peak |
| micro-jet (shoulders, upper arms, wrists, upper/lower back, thighs, calves, heels) | 40 | 40 N | 5 ms | jet gas 600 N·s (shared) |

When a step would overdraw a tank all thrusters on it are scaled so the tank lands at
exactly 0; an empty tank forces u_max = 0.

**Sensors (suitsim/sensors).** One IMU per segment (15; accel σ 0.05 m/s², gyro σ 0.01 rad/s,
2 ms delay), downward radar altimeter on the pelvis (2 % range noise, 5 ms delay, ignores
the wearer's own geoms), foot pressure (5 %), joint encoders (0.002 rad), thruster telemetry.

**Estimator (suitsim/estimation).** Sensor-only, 500 Hz: gyro-integrated torso attitude
with a quasi-static gravity correction (tilt only); kinematics on a private `MjData` from
estimated attitude + encoders (CoM, thruster sites, altimeter pose); CoM acceleration as the
mass-weighted mean of all 15 IMU specific forces (exactly −g in free fall, independent of
attitude error); a 2-state Kalman filter for CoM height / vertical velocity fed by the
altimeter (slant-range → vertical via kinematics, gated, with re-acquisition onto new
surfaces such as the platform); free-fall latch from torso **or** CoM specific force (robust
to spin); horizontal velocity by integration + zero-velocity updates when standing.

**Allocation (suitsim/control/allocation.py, fast_allocation.py).** Every 2 ms the desired
CoM wrench [F; τ] is mapped to thruster forces by a bounded, class-regularised least squares
`min ||W(A S v − w)||² + ρ Σ λ_i v_i²,  0 ≤ v ≤ u_max/s` (torque rows weighted 1/0.3 m; λ makes
pods carry vertical force and well-levered jets carry torque), solved exactly by a
warm-started primal active-set method (same optimum as scipy BVLS, typically a few 44×44
solves). Authority at the nominal pose: rank 6; pure torque roll ±27 N·m, pitch −226/+182 N·m,
yaw ±65 N·m; with Fz = mg the pods add differential roll, ±225 N·m. The pods sit ≈3.6 cm
behind the whole-body CoM, so pod thrust creates a pitch torque the jets must cancel.

**Controllers (suitsim/control).** All read only estimator outputs (never ground truth).
* *T1 landing* (`baseline.py`): quadrotor-style — free-fall latch → thrust-axis attitude PD
  with a √-law planner and yaw-rate damping → vertical braking profile
  `v_ref(h) = −√(v_t² + 2 a (h − h_t))` started as late as possible (v_t = 1.2 m/s at 0.3 m,
  a = 9 m/s²) with nominal-mass feed-forward → horizontal damping by tilting the thrust axis
  (≤10°) → thrust cut at touchdown, airbags if a hard or tilted impact is predicted.
* *T2 jump* (`jump.py`): stand/settle → crouch → leg push + pod boost until the predicted
  ballistic apex clears the platform by 0.6 m → ballistic flight with jet-only horizontal
  guidance → T1-style braking relative to the platform top with edge/overhang guards and
  an altitude fusion that adopts the platform surface.
* *T3 balance* (`balance.py`): linear inverted pendulum / instantaneous capture point;
  the feet supply the CMP while it stays inside the (shrunk) support polygon, the remainder
  is requested from the thrusters as a moment about the feet.

**Residual RL (suitsim/control/residual_policy.py, scripts/train.py).** SB3 PPO, MLP
128×128, acts at 50 Hz with a 6-D residual wrench (±300/300/600 N, ±60/60/30 N·m) added to
the baseline wrench (a = 0 reproduces the baseline), observation normalisation, shaped
reward (dense gas penalty 0.004 /N·s, impact-speed and margin penalties near the safety
limits). Trained on T1 with full randomisation on seeds disjoint from the test seeds.
"""

T1_FAILURE_MD = """
**Failure analysis.** Every T1 failure is a *horizontal* impact speed above 1 m/s; vertical
speed and tilt stay inside the limits in all 1000 episodes and nothing but a foot ever
touches first (no airbags were needed). The failures concentrate in short drops (≈1–1.7 m)
with large initial tilt and spin: the body has ~0.5 s of fall, the attitude loop spends it
righting the thrust axis, and the horizontal velocity created by the tilted braking thrust
cannot be damped before touchdown — horizontal damping is only available by tilting the
thrust axis (≤10°), and the jets' direct lateral authority is small (2 side jets per
direction at 40 N). Taller drops (> 2 m) are 100 % in both controllers.
"""

RL_MD = """
**Training.** `ppo_t1_b` (4 envs, log σ₀ = −1.2) ran 450k steps; `ppo_t1_c` continued from its
400k checkpoint for another 450k steps; the selected policy is the best periodic evaluation
of `ppo_t1_c` (850k steps, `results/policies/residual_t1/model.zip` + `vecnormalize.pkl`).
Periodic evaluations use 50 held-out episodes (seeds ≥ 100000); the baseline on those same
episodes is shown dotted. The curve is still falling at the end (≈ −5 % total gas vs
baseline on the eval seeds) — far from converged: the spec budget is 20 M steps, which at
the measured 130–152 steps/s on this 4-core container is **37–43 h** (≈ 9–11 h on 16 cores
with linear scaling). The residual policy trades margin for gas: it brakes later and lands
faster (mean |vz| up ~0.25 m/s, p95 still below 2 m/s), and its 1000-episode success is
statistically indistinguishable from the baseline.
"""

T3_INTRO_MD = """
The wearer stands on a 0.3 m wide, 1 m high beam and receives a horizontal push on the torso
(200 N for 0.2 s = 40 N·s, uniformly random direction and onset 0.5–1.5 s); success = both
feet still on the beam, pelvis high and torso tilt < 30° at the end of 4 s. Stress tests
re-run the same controller with `--set task.push_force=400|500` (80 / 100 N·s). The side-jet
variant (`configs/task_push_side_jets.yaml` = task_push + the 4 jets of
`configs/suit_variant_side_jets.yaml`, 48 thrusters) runs the same controller unchanged — the
allocator picks up the extra columns automatically.
"""

T4_MD = """
No evaluated episode overdrew a budget (the thruster bank clips at an empty tank, so "within"
means the controller never ran a tank dry before the end of the episode; a tank that hits
exactly 0 is counted as within but is flagged by the "max jets frac" column). Pods are far
from their limit in every task (T1 p95 ≈ 60 %), the fans use a negligible fraction of the
battery, and **jet gas is the binding resource**: T2 averages ~80 % of the jet tank and T3
pushes above ~400 N empty it. The jets spend a large part of their gas cancelling the pod
pitch moment (pods 3.6 cm behind the CoM) — see section 9.2.
"""

VIDEO_MD = """
Each evaluation re-simulates its 5 worst and 5 best episodes (ranked by success, then a
badness score: impact speeds and tilt relative to the limits for landing tasks; final tilt,
lateral CoM offset and jet gas for push recovery) and renders them headless at 640×480, real
time. Each worst #0 clip also has a time-series PNG (height, speeds, tilt, thrust by class,
budgets). Videos are in `videos/<evaluation>/`.
"""

REC_INTRO_MD = """
Two cheap changes to the default layout, each tested with the *unchanged* baseline
controllers (the allocator adapts to the thruster set from the YAML).
"""

SIDE_JET_MD = """
The default suit has only two lateral jets per direction (upper back + thigh side jets), so
its sideways authority about the feet is ≈1.25 m/s² versus 4–5 m/s² sagittally; lateral pushes
are exactly where the 500 N failures sit. Adding four 40 N lateral jets (one per side at the
shoulders, one per side at the hips; +4 jets, ≈ +0.2 kg) closes most of that gap at the same
jet tank. **Recommendation: add the 4 side jets** (48 thrusters). It costs nothing at nominal
loads and is the single largest robustness gain found for T3.
"""

PODS_MD = """
**Recommendation: mount the pods ≈3.6 cm further forward** (or shift the hip-pod mass
equivalently) so the pod thrust line passes through the whole-body CoM in the landing
posture; the jets then no longer have to hold a steady counter-pitch moment while the pods
brake. Since the CoM moves with posture (and differs between wearers), an adjustable pod
mount (±3 cm) is the practical form of this change.
"""

REC_SUMMARY_MD = """
### 9.3 Summary of layout recommendations

1. **+4 lateral micro-jets** (shoulders and hips, 40 N each) — 48 thrusters; raises lateral
   push tolerance (section 9.1). No change needed elsewhere.
2. **Pods forward onto the CoM** (adjustable mount) — removes the static pitch moment that
   the jets cancel during every braking burn (section 9.2).
3. Jet tank is the binding budget in T2/T3; if mass allows, shift capacity from the pod tank
   (T1 p95 use ≈ 60 % of 2500 N·s) to the jet tank (600 N·s) rather than adding gas overall.
"""

LIMITS_MD = """
* **RL budget.** The residual policy saw 850k steps (4 % of the 20 M-step spec). The
  projected 20 M run is 37–43 h on this 4-core box (≈9–11 h on 16 cores). The training curve
  was still improving; the residual is the right place to recover the low-drop horizontal
  failures, but it has not done so yet.
* **Airbag contact accounting.** A deployed airbag touching the ground is attributed to its
  carrier segment (torso / pelvis) and therefore counts as a "bad contact" in the env's
  bookkeeping. No T1 episode deployed airbags, so no reported number is affected, but the
  airbag path is effectively untested end-to-end and the metric must be fixed (count airbag
  geoms separately) before airbag-dependent scenarios are evaluated.
* **Estimator tilt residual.** In free fall and under thrust the attitude is gyro-only (the
  gravity correction needs quasi-static conditions), so the ±2° initial alignment error and
  gyro noise persist through the drop; standing, the correction leaves ~0.5–1° (postural sway
  read as tilt). This sets a floor on horizontal-velocity accuracy (1° ≈ 0.17 m/s² of
  mis-attributed thrust) and is a plausible contributor to the horizontal-speed failures.
* **EMG** is a config stub (`emg.enabled: false`): no wearer-intent input is modelled; the
  human is a passive servo-held body.
* **No MyoSuite / musculoskeletal model**: joints are torque-limited PD servos, not muscles;
  landing loads on the joints are not evaluated.
* T2 assumes the controller knows the platform position (`jump.mission_knows_platform`).
* T3's beam is static and the push is a pure force on the torso CoM.
* Statistics: T2/T3 at n = 200 give 100 % with a 98.1 % lower Wilson bound; T1 at n = 1000
  resolves ≈ ±0.4 % around 99.5 %.

**Next steps:** (1) 20 M-step residual PPO on 16 cores, with low-drop/high-spin curriculum;
(2) adopt the side-jet + pods-forward layout as the new default and re-run T1–T3;
(3) separate airbag contacts in the bookkeeping and add an airbag-required scenario;
(4) jet-tank sizing study (T2 is the limiting mission); (5) EMG/intent and a musculoskeletal
human (MyoSuite) for joint-load assessment.
"""

REPRO_MD = """
```bash
python scripts/evaluate.py --task configs/task_drop.yaml --controller baseline --episodes 1000 --seed 0 --workers 4 --out runs/eval_baseline_t1 --videos 5
python scripts/evaluate.py --task configs/task_drop.yaml --controller residual --policy results/policies/residual_t1/model.zip --episodes 1000 --seed 0 --workers 4 --out runs/eval_residual_t1 --videos 5
python scripts/evaluate.py --task configs/task_jump.yaml --episodes 200 --workers 4 --out runs/eval_t2
python scripts/evaluate.py --task configs/task_push.yaml --episodes 200 --workers 4 --out runs/eval_t3_final
python scripts/evaluate.py --task configs/task_push_side_jets.yaml --episodes 200 --workers 4 --out runs/eval_t3_side_jets
python scripts/evaluate.py --task configs/task_push.yaml --episodes 100 --set task.push_force=500 --workers 4 --out runs/stress_t3_500N_default      # also 400
python scripts/evaluate.py --task configs/task_push_side_jets.yaml --episodes 100 --set task.push_force=500 --workers 4 --out runs/stress_t3_500N_side_jets
python scripts/evaluate.py --task configs/task_drop_side_jets.yaml --episodes 200 --workers 4 --out runs/eval_t1_side_jets
python scripts/evaluate.py --task configs/task_drop_pods_fwd.yaml --episodes 200 --workers 4 --out runs/eval_t1_pods_fwd
python scripts/make_report.py
```
Summaries of every evaluation are copied to `results/m7_*_summary.json`.
"""

if __name__ == "__main__":
    sys.exit(main())
