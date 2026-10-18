# mobility-suit-sim

Physics simulator and control stack for a **thruster-based wearable mobility suit**: a
91 kg human in a 9 kg suit with 2 lift pods, 2 ducted fans, 40 micro-jets and 3 airbags,
simulated in MuJoCo with realistic sensors, a sensor-only state estimator, a constrained
thrust allocator, model-based controllers and a residual PPO policy.

Tasks (all fully domain-randomised: human mass, thrust, thruster lag, sensor noise, friction):

| task | scenario | config |
|---|---|---|
| T1 | drop landing from 1–5 m, ≤30° tilt, ≤2 rad/s spin per axis | `configs/task_drop.yaml` |
| T2 | boosted jump onto a 2.5 m platform 1.3–1.7 m away | `configs/task_jump.yaml` |
| T3 | push recovery (200 N × 0.2 s, any direction) on a 0.3 m beam | `configs/task_push.yaml` |
| T4 | consumable budgets: pod gas 2500 N·s, jet gas 600 N·s, battery 150 Wh | `configs/suit_default.yaml` |

## Results

Full report with methods, failure analysis, histograms, training curves and videos:
**[reports/M7_report.md](reports/M7_report.md)**.

| evaluation | episodes | success (95 % Wilson CI) | gas used, mean (pods / jets) [N·s] |
|---|---|---|---|
| T1 drop — baseline controller | 1000 | 99.5 % [98.8, 99.8] | 1112 / 298 |
| T1 drop — residual PPO | 1000 | 99.4 % [98.7, 99.7] | 1015 / 307 (−6.3 % total) |
| T2 boosted jump | 200 | 100 % [98.1, 100] | 1130 / 482 |
| T3 push 200 N | 200 | 100 % [98.1, 100] | 0 / 81 |
| T3 push 200 N — +4 side-jet suit variant | 200 | 100 % [98.1, 100] | 0 / 55 |
| T3 stress 400 N — default / side-jet suit | 100 + 100 | 88 % [80, 93] / 100 % [96, 100] | 143 / 373 vs 25 / 220 |
| T3 stress 500 N — default / side-jet suit | 100 + 100 | 51 % [41, 61] / 90 % [83, 95] | 172 / 444 vs 80 / 347 |
| T1 drop — pods 3.6 cm forward (seeds 0–199) | 200 | 99.5 % [97.2, 99.9] | 1064 / 276 (−5.9 % total vs default) |

All evaluated episodes stayed within every budget (T4); jet gas (600 N·s) is the binding one.
Recommended layout changes: +4 lateral micro-jets and lift pods moved ≈3.6 cm forward onto
the CoM (report section 9).

## Install

Python ≥ 3.11.

```bash
pip install -e .[dev]          # mujoco, gymnasium, stable-baselines3, numpy, scipy, matplotlib, imageio(-ffmpeg), pytest
```

Headless rendering (videos) needs an offscreen OpenGL backend. On Debian/Ubuntu:

```bash
sudo apt-get install -y libegl1 libosmesa6 libgl1-mesa-dri
```

`suitsim.viz.gl.ensure_gl()` probes `egl` then `osmesa` and exports the first that works as
`MUJOCO_GL`; set `MUJOCO_GL=egl` (GPU) or `MUJOCO_GL=osmesa` (CPU software rendering)
yourself to skip the probe. Simulation and tests that do not render need no GL at all.

## Quick start

```bash
python -m pytest -q                                   # full test suite (~134 tests)

# T1 baseline statistics (no videos)
python scripts/run_baseline.py --episodes 1000 --seed 0 --workers 4 --out runs/baseline_t1

# full evaluation folder: summary.json/.md, results.csv/.jsonl, plots/, 5 worst + 5 best videos
python scripts/evaluate.py --task configs/task_drop.yaml --controller baseline \
    --episodes 1000 --seed 0 --workers 4 --out runs/eval_baseline_t1 --videos 5
python scripts/evaluate.py --task configs/task_drop.yaml --controller residual \
    --policy results/policies/residual_t1/model.zip --episodes 1000 --workers 4 --out runs/eval_residual_t1
python scripts/evaluate.py --task configs/task_jump.yaml --episodes 200 --workers 4 --out runs/eval_t2
python scripts/evaluate.py --task configs/task_push.yaml --episodes 200 --workers 4 --out runs/eval_t3
python scripts/evaluate.py --task configs/task_push.yaml --set task.push_force=500 --episodes 100 --out runs/stress_500N
python scripts/evaluate.py --quick                    # 20-episode smoke test

# one episode -> mp4 + time-series PNG + info JSON
python scripts/render_episode.py --task configs/task_drop.yaml --controller baseline --seed 42 \
    --out runs/render/drop_seed42.mp4

# residual PPO on T1 (--quick: 4 envs, 200k steps; the 20 M-step spec run is ~40 h on 4 cores)
python scripts/train.py --config configs/train_ppo.yaml --quick --out runs/ppo_quick

# rebuild reports/M7_report.md (+ figures, videos, results/m7_*_summary.json) from runs/
python scripts/make_report.py
```

## Repository layout

```
configs/                 YAML configs (all physical parameters, gains, task definitions)
  suit_default.yaml        human + suit model, thrusters, budgets, sensors, estimator, allocation
  task_drop.yaml           T1 + baseline landing controller + residual scale + randomisation
  task_jump.yaml           T2 (base: task_drop.yaml) + jump controller
  task_push.yaml           T3 (base: task_drop.yaml) + balance controller
  train_ppo.yaml           PPO hyper-parameters and training-only reward overrides
  suit_variant_*.yaml      thruster layout variants (side jets, pods forward)
  task_*_side_jets.yaml, task_drop_pods_fwd.yaml   tasks with those variants applied
suitsim/
  interfaces.py            shared contract: names, keys, config loading, thruster specs, metadata
  model/                   procedural MJCF builder, posture, airbags, domain randomisation
  actuators/               thruster bank (lag, clamps, shared budgets), MuJoCo wrench application
  sensors/                 IMUs, altimeter ray, foot pressure, encoders, telemetry (noise + delay)
  estimation/              sensor-only state estimator + kinematics
  control/                 allocation (BVLS / active set), T1 baseline, T2 jump, T3 balance, residual policy
  envs/                    Gymnasium envs: base loop, drop_landing, boosted_jump, push_recovery
  eval/                    batch runner, metrics (Wilson CIs, failure modes, budgets), reports
  viz/                     headless GL selection, video recorder (thrust plumes), time-series plots
scripts/                 run_baseline, evaluate, render_episode, train, make_report
tests/                   pytest suite
assets/human_suit.xml    MJCF generated from the default config (reference copy)
results/                 tracked summaries (m4/m5/m6/m7_*_summary.json) and the trained policy
reports/                 M7 report, figures and selected videos (self-contained)
runs/                    scratch output of every script (git-ignored)
```

## Config rules

* **Every physical parameter, gain and limit lives in YAML**; code only holds cosmetic
  values (colours, camera). Comment units next to each value.
* Configs chain with a single `base:` key (path relative to the file); the child is
  deep-merged over its base. Dict entries merge key by key, lists are replaced. Thrusters are
  a name-keyed dict, so a variant can move one (`pod_l: {pos: ...}`) or append new ones.
* Any script takes `--set a.b.c=value` (value parsed as YAML) for one-off overrides, e.g.
  `--set task.push_force=500` or `--set task.fixed.drop_height=3.0`.
* Controllers consume only sensor / estimator dicts, never MuJoCo ground truth (ground
  truth is used for metrics only).
* Every run writes `config.yaml` + `meta.json` (git hash, config hash, argv) next to its
  outputs; evaluation seeds are `seed … seed+episodes−1` and reproduce bit-for-bit.
