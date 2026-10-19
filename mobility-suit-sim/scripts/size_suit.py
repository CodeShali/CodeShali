#!/usr/bin/env python
"""Hardware sizing calculator: mission requirements -> component specs and suit mass.

    python scripts/size_suit.py                          # all architectures, text tables
    python scripts/size_suit.py --arch B                 # one architecture's budget
    python scripts/size_suit.py --json runs/sizing.json  # full results as JSON
    python scripts/size_suit.py --markdown runs/sizing.md
    python scripts/size_suit.py --update-doc             # refresh the generated section of
                                                         #   docs/hardware_sizing.md
    python scripts/size_suit.py --bench                  # also time the simulator's allocator

Assumptions (all ranges [optimistic, nominal, conservative]) are in configs/sizing.yaml.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from suitsim.sizing import load_sizing_config, run  # noqa: E402
from suitsim.sizing.report import render, update_doc  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "sizing.yaml"))
    ap.add_argument("--arch", action="append", help="architecture key (A/B/C/D); repeatable")
    ap.add_argument("--json", help="write full results JSON here")
    ap.add_argument("--markdown", help="write the Markdown tables here")
    ap.add_argument("--update-doc", nargs="?", const=str(ROOT / "docs" / "hardware_sizing.md"),
                    help="rewrite the generated section of docs/hardware_sizing.md (or PATH)")
    ap.add_argument("--bench", action="store_true", help="time the simulator's allocator here")
    ap.add_argument("--quiet", action="store_true", help="no text tables on stdout")
    a = ap.parse_args(argv)

    cfg = load_sizing_config(a.config)
    bad = [k for k in (a.arch or []) if k not in cfg["architectures"]]
    if bad:
        ap.error(f"unknown architecture(s) {bad}; known: {list(cfg['architectures'])}")
    # all architectures are always sized (cross-references); --arch filters the printout
    res = run(cfg, bench=a.bench or bool(a.update_doc))
    if not a.quiet:
        print(render(res, cfg, a.arch, md=False))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1, default=str))
        print(f"wrote {a.json}")
    if a.markdown or a.update_doc:
        md = render(res, cfg, a.arch, md=True)
        if a.markdown:
            Path(a.markdown).parent.mkdir(parents=True, exist_ok=True)
            Path(a.markdown).write_text(md)
            print(f"wrote {a.markdown}")
        if a.update_doc:
            if not update_doc(a.update_doc, md):
                print(f"no generated-section markers in {a.update_doc}", file=sys.stderr)
                return 1
            print(f"updated {a.update_doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
