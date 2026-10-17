#!/usr/bin/env python
"""Render one episode (single seed) to mp4 + a time-series plot.

    python scripts/render_episode.py --task configs/task_drop.yaml --controller baseline \
        --seed 42 --out runs/render/drop_seed42.mp4
    python scripts/render_episode.py --controller residual --policy runs/ppo/model.zip --seed 7

Writes <out>.mp4, <out>.png (height / velocity / tilt / thrust / budgets) and <out>.json
(episode info).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from suitsim.viz.gl import ensure_gl  # noqa: E402

ensure_gl()

from suitsim.eval.batch import EnvSpec, parse_overrides  # noqa: E402
from suitsim.eval.metrics import primary_failure  # noqa: E402
from suitsim.eval.report import write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", default="configs/task_drop.yaml")
    ap.add_argument("--controller", default="baseline", choices=["baseline", "residual", "none"])
    ap.add_argument("--policy", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output .mp4 (or .gif)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--camera", default="track", help="model camera name (fallback: follow)")
    ap.add_argument("--no-plumes", action="store_true")
    ap.add_argument("--no-shadows", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL")
    ap.add_argument("--factory", default=None, help="env factory 'module:callable'")
    args = ap.parse_args(argv)

    from suitsim.viz.plots import plot_episode
    from suitsim.viz.render import EpisodeRecorder

    spec = EnvSpec(task_cfg=args.task, overrides=parse_overrides(args.set),
                   controller=args.controller, policy_path=args.policy, factory=args.factory)
    cfg = spec.config()
    task = (cfg.get("task") or {}).get("name", "task")
    out = Path(args.out or f"runs/render/{task}_{args.controller}_seed{args.seed}.mp4")
    try:
        env = spec.make_env(log_trajectory=True)
    except TypeError:
        env = spec.make_env()
    rec = EpisodeRecorder(env, spec.make_policy(), width=args.width, height=args.height,
                          fps=args.fps, camera=args.camera, plumes=not args.no_plumes,
                          shadows=not args.no_shadows)
    try:
        res = rec.record(args.seed)
        rec.save_mp4(out)
    finally:
        rec.close()
        env.close()
    plot_episode(res.get("trajectory"), out.with_suffix(".png"), info=res, cfg=cfg,
                 title=f"{task} {args.controller} seed {args.seed} ({primary_failure(res)})")
    res.pop("trajectory", None)
    write_json(res, out.with_suffix(".json"))
    print(json.dumps({k: res.get(k) for k in ("success", "safe", "impact_v_vertical",
                                              "impact_v_horizontal", "tilt_at_impact_deg",
                                              "first_contact", "n_frames")}, default=str))
    print(f"wrote {out}, {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
