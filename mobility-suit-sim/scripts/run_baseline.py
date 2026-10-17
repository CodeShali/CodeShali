#!/usr/bin/env python
"""Run the baseline landing controller on T1 (drop landing) and report success statistics.

    python scripts/run_baseline.py --episodes 1000 --seed 0 --workers 4 --out runs/baseline_t1
    python scripts/run_baseline.py --episodes 10 --fixed-3m          # M4 smoke test

Episode ``i`` uses ``env.reset(seed=seed + i)``.  Writes ``results.jsonl`` (one info dict per
episode), ``summary.json`` and config/meta (``save_run_metadata``) to ``--out``.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from suitsim.interfaces import load_config, save_run_metadata

FIXED_3M = {"drop_height": 3.0, "tilt_deg": 0.0, "angvel": 0.0}

_ENV = None


def _py(x):
    if isinstance(x, dict):
        return {k: _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def _init(cfg: dict, controller: str) -> None:
    global _ENV
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from suitsim.envs import make_env
    _ENV = make_env(cfg["task"]["name"], cfg, controller=controller)


def run_one(seed: int) -> dict:
    env = _ENV
    t0 = time.perf_counter()
    env.reset(seed=int(seed))
    zero = np.zeros(6, dtype=np.float32)
    while True:
        _, _, term, trunc, info = env.step(zero)
        if term or trunc:
            break
    rec = _py(info)
    rec.update(seed=int(seed), wall_time=time.perf_counter() - t0, truncated=bool(trunc))
    return rec


def summarize(results: list[dict], cfg: dict) -> dict:
    n = len(results)
    ok = [r for r in results if r["success"]]
    k = len(ok)
    p = k / max(n, 1)
    z = 1.96   # Wilson 95 % interval
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    reasons = Counter()
    primary = Counter()
    for r in results:
        if r["success"]:
            continue
        fr = r.get("failure_reasons") or ["unknown"]
        reasons.update(fr)
        primary[fr[0]] += 1
    fc = Counter(r["first_contact"] for r in results)

    def stat(key, rs=results):
        v = np.array([r[key] for r in rs if r.get(key) is not None and np.isfinite(r[key])], float)
        if v.size == 0:
            return {}
        return {"mean": float(v.mean()), "p50": float(np.median(v)), "p95": float(np.percentile(v, 95)),
                "max": float(v.max())}

    def binned(param, edges):
        out = {}
        vals = np.array([r["episode_params"][param] for r in results])
        succ = np.array([r["success"] for r in results])
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (vals >= lo) & (vals < hi)
            if m.any():
                out[f"[{lo:g},{hi:g})"] = {"n": int(m.sum()), "success": float(succ[m].mean())}
        return out

    return {
        "episodes": n, "success": k, "success_rate": p, "wilson95": [c - hw, c + hw],
        "failure_reasons_any": dict(reasons), "failure_primary": dict(primary),
        "first_contact": dict(fc),
        "airbags_deployed": int(sum(r["airbags_deployed"] for r in results)),
        "impact_v_vertical": stat("impact_v_vertical"),
        "impact_v_horizontal": stat("impact_v_horizontal"),
        "tilt_at_impact_deg": stat("tilt_at_impact_deg"),
        "impulse_gas_pods": stat("impulse_gas_pods"), "impulse_gas_jets": stat("impulse_gas_jets"),
        "battery_wh": stat("battery_wh"),
        "budget_pods_cap": cfg["suit"]["budgets"]["impulse_gas_pods"],
        "budget_jets_cap": cfg["suit"]["budgets"]["impulse_gas_jets"],
        "episode_time": stat("episode_time"), "wall_time": stat("wall_time"),
        "by_drop_height": binned("drop_height", [1, 2, 3, 4, 5.0001]),
        "by_tilt_deg": binned("tilt_deg", [0, 10, 20, 30.0001]),
        "by_angvel": binned("angvel", [0, 1, 2, 3, 3.5]),
        "by_mass_scale": binned("mass_scale", [0.9, 0.95, 1.0, 1.05, 1.10001]),
        "by_pod_fmax": binned("pod_fmax_scale_mean", [0.85, 0.9, 0.95, 1.0, 1.05, 1.10, 1.15001]),
    }


def print_summary(s: dict) -> None:
    print(f"\nepisodes {s['episodes']}  success {s['success']}  rate {100 * s['success_rate']:.1f}%"
          f"  (95% CI {100 * s['wilson95'][0]:.1f}-{100 * s['wilson95'][1]:.1f}%)")
    print("failure (primary):", s["failure_primary"], " any:", s["failure_reasons_any"])
    print("first contact:", s["first_contact"], " airbags deployed:", s["airbags_deployed"])
    for k in ("impact_v_vertical", "impact_v_horizontal", "tilt_at_impact_deg", "impulse_gas_pods",
              "impulse_gas_jets", "battery_wh", "episode_time", "wall_time"):
        print(f"  {k:22s}", {a: round(b, 3) for a, b in s[k].items()})
    for k in ("by_drop_height", "by_tilt_deg", "by_angvel", "by_mass_scale", "by_pod_fmax"):
        print(f"  {k:15s}", {a: f"{100 * b['success']:.0f}% (n={b['n']})" for a, b in s[k].items()})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="task_drop.yaml")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--controller", default="baseline", choices=["baseline", "none"])
    ap.add_argument("--fixed-3m", action="store_true", help="fixed 3 m upright drop, no DR (M4)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="config override, e.g. --set baseline.target_speed=1.2")
    ap.add_argument("--out", default=None, help="output dir (default runs/baseline_<time>)")
    a = ap.parse_args(argv)

    import yaml
    ov: dict = {}
    for item in a.set:
        key, _, val = item.partition("=")
        d = ov
        parts = key.split(".")
        for q in parts[:-1]:
            d = d.setdefault(q, {})
        d[parts[-1]] = yaml.safe_load(val)
    if a.fixed_3m:
        ov.setdefault("task", {})["fixed"] = dict(FIXED_3M)
        ov.setdefault("randomization", {})["enabled"] = False
    cfg = load_config(a.config, ov)
    out = Path(a.out or f"runs/baseline_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)

    seeds = list(range(a.seed, a.seed + a.episodes))
    t0 = time.time()
    results = []
    with open(out / "results.jsonl", "w") as f:
        if a.workers <= 1:
            _init(cfg, a.controller)
            it = map(run_one, seeds)
            pool = None
        else:
            pool = mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(cfg, a.controller))
            it = pool.imap(run_one, seeds, chunksize=4)
        for i, r in enumerate(it):
            results.append(r)
            f.write(json.dumps(r) + "\n")
            if (i + 1) % 50 == 0:
                sr = np.mean([x["success"] for x in results])
                print(f"  {i + 1}/{len(seeds)}  success {100 * sr:.1f}%  ({time.time() - t0:.0f}s)",
                      flush=True)
        if pool is not None:
            pool.close()
            pool.join()
    wall = time.time() - t0
    s = summarize(results, cfg)
    s["total_wall_s"] = wall
    s["sim_s_per_wall_s"] = float(sum(r["episode_time"] for r in results) / wall)
    with open(out / "summary.json", "w") as f:
        json.dump(s, f, indent=2)
    save_run_metadata(out, cfg, {"script": "run_baseline.py", "argv": sys.argv[1:],
                                 "episodes": a.episodes, "seed": a.seed,
                                 "controller": a.controller, "success_rate": s["success_rate"]})
    print_summary(s)
    print(f"wall {wall:.0f}s  ({s['sim_s_per_wall_s']:.2f} sim-s per wall-s over {a.workers} workers)"
          f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
