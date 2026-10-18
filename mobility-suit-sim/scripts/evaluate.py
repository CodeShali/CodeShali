#!/usr/bin/env python
"""Batch-evaluate a controller on a task and write a full evaluation folder.

    python scripts/evaluate.py --task configs/task_drop.yaml --controller baseline \
        --episodes 1000 --seed 0 --workers 4 --out runs/eval_baseline_drop

    python scripts/evaluate.py --task configs/task_drop.yaml --controller residual \
        --policy runs/ppo_drop/model.zip --episodes 1000 --workers 4 --out runs/eval_ppo

Outputs in --out:
    summary.json / summary.md     aggregate metrics (Wilson CIs, failure modes, budgets)
    results.csv / results.jsonl   one row per episode (seed, info keys, episode params)
    plots/*.png                   histograms (impact speed, tilt, budgets, failure modes)
    config.yaml / meta.json       config + git hash (save_run_metadata)
    videos/{worst,best}_<i>_seed<s>.mp4 + .png   the k worst and k best episodes (always)

Seeds are ``seed, seed+1, ..., seed+episodes-1``. ``--quick`` = 20 episodes, 2+2 small
videos (smoke test). ``--factory module:callable`` swaps the env constructor (e.g.
``suitsim.eval.fake_env:make_fake_env`` before the real envs exist).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from suitsim.viz.gl import ensure_gl  # noqa: E402

ensure_gl()  # select a headless GL backend before anything imports mujoco

from suitsim.eval.batch import EnvSpec, parse_overrides, results_equal, run_episodes  # noqa: E402
from suitsim.eval.metrics import outcome_label, primary_failure, summarize, worst_best  # noqa: E402
from suitsim.eval.report import write_eval_outputs, write_json  # noqa: E402
from suitsim.interfaces import save_run_metadata  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", default="configs/task_drop.yaml", help="task YAML config")
    ap.add_argument("--controller", default="baseline", choices=["baseline", "residual", "none"])
    ap.add_argument("--policy", default=None, help="SB3 model .zip (residual controller)")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0, help="first seed")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=None, help="output dir (default runs/eval_<task>_<ctrl>)")
    ap.add_argument("--videos", type=int, default=5, help="k worst + k best videos (0=off)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=None, help="video fps (default cfg/50)")
    ap.add_argument("--no-plumes", action="store_true")
    ap.add_argument("--no-shadows", action="store_true")
    ap.add_argument("--quick", action="store_true", help="<=20 episodes, 2+2 small videos, no shadows")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="config override, e.g. --set task.fixed.drop_height=3.0")
    ap.add_argument("--factory", default=None, help="env factory 'module:callable'")
    ap.add_argument("--max-steps", type=int, default=100_000)
    return ap


def rank_key(cfg: dict, limits: dict):
    """Sort key (higher = better) for the worst/best video pick.

    Landing tasks use :func:`suitsim.eval.metrics.episode_score` (impact speed / tilt).
    Push recovery never 'lands' (impact speeds are 0 and tilt_at_impact is constant), so
    ties would fall back to seed order; rank it by the end state instead: final torso tilt,
    lateral CoM offset relative to the half beam width and jet gas used.
    """
    from suitsim.eval.metrics import episode_score
    task = (cfg.get("task") or {})
    if task.get("name") != "push_recovery":
        return lambda r: episode_score(r, limits)
    max_tilt = float((cfg.get("push_success") or {}).get("max_tilt_deg", 30.0))
    half_w = 0.5 * float(task.get("beam_width", 0.3))
    jets_cap = float(((cfg.get("suit") or {}).get("budgets") or {}).get("impulse_gas_jets", 600.0))

    def key(r: dict):
        bad = (abs(float(r.get("final_tilt_deg") or 0.0)) / max_tilt
               + abs(float(r.get("final_com_offset_y") or 0.0)) / half_w
               + float(r.get("impulse_gas_jets") or 0.0) / jets_cap)
        return (int(bool(r.get("success"))), int(bool(r.get("safe"))), -bad)
    return key


def render_ranked(spec: EnvSpec, ranked: list[tuple[str, int, dict]], out_dir: Path,
                  width: int, height: int, fps: float | None, plumes: bool, cfg: dict,
                  shadows: bool = True) -> list:
    """Re-run and render each (label, rank, result); also writes a time-series PNG."""
    from suitsim.viz.plots import plot_episode
    from suitsim.viz.render import EpisodeRecorder

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        env = spec.make_env(log_trajectory=True)
    except TypeError:            # env without trajectory logging support
        env = spec.make_env()
    policy = spec.make_policy()
    rec = EpisodeRecorder(env, policy, width=width, height=height, fps=fps, plumes=plumes,
                          shadows=shadows)
    made = []
    try:
        for label, i, r in ranked:
            seed = int(r["seed"])
            stem = f"{label}_{i}_seed{seed}"
            t0 = time.perf_counter()
            res = rec.record(seed)
            mp4 = rec.save_mp4(out_dir / f"{stem}.mp4")
            png = plot_episode(res.get("trajectory"), out_dir / f"{stem}.png", info=res, cfg=cfg,
                               title=f"{spec.controller} {label} #{i} seed {seed} ({outcome_label(res)})")
            same = results_equal([r], [res], tol=1e-6)
            made.append({"label": label, "rank": i, "seed": seed, "video": mp4.name,
                         "plot": png.name, "frames": res["n_frames"], "deterministic": same,
                         "success": res.get("success"), "failure_mode": primary_failure(res)})
            print(f"  {stem}: {res['n_frames']} frames in {time.perf_counter() - t0:.1f}s"
                  + ("" if same else "  [WARNING: re-run differs from batch result]"), flush=True)
    finally:
        rec.close()
        if hasattr(env, "close"):
            env.close()
    return made


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.quick:
        args.episodes = min(args.episodes, 20)
        args.videos = min(args.videos, 2)
        args.width, args.height = min(args.width, 320), min(args.height, 240)
        args.no_shadows = True        # shadow mapping is ~70 % of software-GL frame time
    if args.controller == "residual" and not args.policy:
        print("--controller residual needs --policy", file=sys.stderr)
        return 2
    spec = EnvSpec(task_cfg=args.task, overrides=parse_overrides(args.set),
                   controller=args.controller, policy_path=args.policy, factory=args.factory)
    cfg = spec.config()
    task = (cfg.get("task") or {}).get("name", Path(args.task).stem)
    out = Path(args.out or f"runs/eval_{task}_{args.controller}")
    out.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seed, args.seed + args.episodes))
    print(f"[evaluate] {task} / {args.controller}: {len(seeds)} episodes, "
          f"{args.workers} worker(s) -> {out}", flush=True)
    t0 = time.perf_counter()
    results = run_episodes(spec, None, seeds, n_workers=args.workers, max_steps=args.max_steps,
                           progress=True)
    wall = time.perf_counter() - t0
    errors = [r for r in results if "error" in r]
    if errors:
        print(f"[evaluate] {len(errors)} episode(s) crashed, e.g. seed {errors[0]['seed']}: "
              f"{errors[0]['error']}", file=sys.stderr)

    summary = summarize(results, cfg)
    summary.update(controller=args.controller, policy=args.policy, seeds=[seeds[0], seeds[-1]],
                   wall_time_s=wall, n_errors=len(errors))
    save_run_metadata(out, cfg, extra={"script": "scripts/evaluate.py", "argv": sys.argv[1:],
                                       "controller": args.controller, "policy": args.policy,
                                       "seeds": [seeds[0], seeds[-1]], "workers": args.workers})
    write_eval_outputs(results, summary, out, title=f"{task} — {args.controller}")

    if args.videos > 0:
        ok = [r for r in results if "error" not in r]
        worst, best = worst_best(ok, k=args.videos, key=rank_key(cfg, summary["limits"]))
        ranked = [("worst", i, r) for i, r in enumerate(worst)] + \
                 [("best", i, r) for i, r in enumerate(best)]
        print(f"[evaluate] rendering {len(ranked)} videos", flush=True)
        try:
            vids = render_ranked(spec, ranked, out / "videos", args.width, args.height,
                                 args.fps, not args.no_plumes, cfg,
                                 shadows=not args.no_shadows)
            write_json(vids, out / "videos" / "index.json")
            summary["videos"] = vids
            write_json(summary, out / "summary.json")
        except Exception as e:  # rendering must not lose the evaluation itself
            print(f"[evaluate] video rendering failed: {type(e).__name__}: {e}", file=sys.stderr)
            summary["videos_error"] = f"{type(e).__name__}: {e}"
            write_json(summary, out / "summary.json")

    s = summary
    print(json.dumps({"success": s["success"], "safe": s["safe"],
                      "failures": s["failures"]["primary"],
                      "within_budgets": s["budget_usage"]["all_within"]["rate"],
                      "wall_time_s": round(wall, 1)}, indent=1))
    print(f"[evaluate] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
