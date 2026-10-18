#!/usr/bin/env python
"""Train the residual PPO policy on top of the baseline landing controller (M5).

    python scripts/train.py --config configs/train_ppo.yaml --quick --out runs/ppo_quick
    python scripts/train.py --config configs/train_ppo.yaml --total-timesteps 2000000 \
        --n-envs 4 --seed 0 --out runs/ppo_2m

Outputs in --out:
    model.zip + vecnormalize.pkl       final policy (loadable by scripts/evaluate.py
                                       --controller residual --policy <out>/model.zip)
    best/model.zip + vecnormalize.pkl  best periodic-eval checkpoint (highest eval score)
    checkpoints/                       periodic checkpoints (+ VecNormalize stats)
    eval.csv                           periodic deterministic eval (success, impulse, ...)
    eval_baseline.json                 the baseline on the same eval seeds (reference)
    progress.csv                       SB3 training log (+ rollout safe rate / impulse)
    config.yaml / meta.json            train + task config, git hash, timing (save_run_metadata)
    timing.json                        throughput, s per 100k steps, projected wall time

Envs run in a ``SubprocVecEnv`` (spawn); env ``i`` is seeded ``seed + i`` (SB3 convention),
eval episodes use seeds ``eval_seed0 + k`` (disjoint from the 0..999 test seeds).
``--quick`` uses the ``quick:`` section of the config (spec: 4 envs, 200k steps).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from suitsim.interfaces import PROJECT_ROOT, deep_merge, load_config, save_run_metadata  # noqa: E402

VECNORM = "vecnormalize.pkl"


# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------
def parse_overrides(items: list[str]) -> dict:
    import yaml
    out: dict = {}
    for it in items or []:
        key, _, val = it.partition("=")
        d = out
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(val)
    return out


def _resolve(path: str | Path, rel_to: Path | None = None) -> Path:
    p = Path(path)
    for cand in ([p] + ([rel_to / p] if rel_to else []) + [PROJECT_ROOT / p,
                                                            PROJECT_ROOT / "configs" / p]):
        if cand.exists():
            return cand.resolve()
    return p


def build_configs(args) -> tuple[dict, dict]:
    """(train_cfg, task_cfg) after --quick / CLI / --set overrides."""
    import yaml
    tpath = _resolve(args.config)
    with open(tpath) as f:
        tcfg = yaml.safe_load(f) or {}
    if args.quick:
        tcfg = deep_merge(tcfg, tcfg.get("quick") or {})
    tcfg = deep_merge(tcfg, parse_overrides(args.train_set))
    if args.total_timesteps is not None:
        tcfg["total_timesteps"] = int(args.total_timesteps)
    if args.n_envs is not None:
        tcfg["n_envs"] = int(args.n_envs)
    if args.seed is not None:
        tcfg["seed"] = int(args.seed)
    if args.eval_freq is not None:
        tcfg["eval_freq"] = int(args.eval_freq)
    if args.eval_episodes is not None:
        tcfg["eval_episodes"] = int(args.eval_episodes)
    task_path = _resolve(tcfg.get("task_config", "task_drop.yaml"), tpath.parent)
    task_over = deep_merge(tcfg.get("task_overrides") or {}, parse_overrides(args.set))
    if "policy_dt" in tcfg:
        task_over = deep_merge({"policy_dt": tcfg["policy_dt"]}, task_over)
    task_cfg = load_config(task_path, task_over)
    tcfg["task_config_path"] = str(task_path)
    return tcfg, task_cfg


# --------------------------------------------------------------------------------------
# env construction (picklable thunks for SubprocVecEnv / spawn)
# --------------------------------------------------------------------------------------
INFO_KEYS = ("safe", "impulse_gas_pods", "impulse_gas_jets", "impact_v_vertical",
             "impact_v_horizontal", "tilt_at_impact_deg")


class _EnvThunk:
    def __init__(self, task_cfg: dict):
        self.task_cfg = task_cfg

    def __call__(self):
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        from stable_baselines3.common.monitor import Monitor
        from suitsim.envs import make_env
        env = make_env(self.task_cfg["task"]["name"], self.task_cfg, controller="residual")
        return Monitor(env, info_keywords=INFO_KEYS)


def make_vec_env(task_cfg: dict, n_envs: int, seed: int, subproc: bool = True):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    thunks = [_EnvThunk(task_cfg) for _ in range(n_envs)]
    venv = (SubprocVecEnv(thunks, start_method="spawn") if subproc and n_envs > 1
            else DummyVecEnv(thunks))
    venv.seed(seed)
    venv.action_space.seed(seed)
    return venv


# --------------------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------------------
def eval_stats(results: list[dict]) -> dict:
    ok = [r for r in results if "error" not in r]
    n = len(results)
    def m(k):
        v = [float(r[k]) for r in ok if r.get(k) is not None and np.isfinite(float(r[k]))]
        return float(np.mean(v)) if v else float("nan")
    succ = sum(bool(r.get("success")) for r in results)
    fails: dict = {}
    for r in ok:
        if not r.get("success"):
            for f in r.get("failure_reasons", []) or ["unknown"]:
                fails[f] = fails.get(f, 0) + 1
    pods, jets = m("impulse_gas_pods"), m("impulse_gas_jets")
    return {"episodes": n, "success_rate": succ / max(n, 1), "n_fail": n - succ,
            "impulse_pods": pods, "impulse_jets": jets, "impulse_total": pods + jets,
            "impact_v_vertical": m("impact_v_vertical"),
            "impact_v_horizontal": m("impact_v_horizontal"),
            "tilt_at_impact_deg": m("tilt_at_impact_deg"),
            "episode_return": m("episode_return"), "n_errors": n - len(ok),
            "failures": fails}


def eval_score(st: dict, ref: dict | None = None) -> float:
    """Model-selection score: success dominates, then total impulse (lower is better)."""
    return 100.0 * st["success_rate"] - 0.01 * st["impulse_total"]


def run_eval(task_cfg: dict, policy_path: str | None, seeds: list[int], workers: int) -> dict:
    from suitsim.eval.batch import EnvSpec, run_episodes
    spec = EnvSpec(cfg=task_cfg, controller="residual" if policy_path else "baseline",
                   policy_path=policy_path)
    return eval_stats(run_episodes(spec, None, seeds, n_workers=workers))


def _save(model, venv, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "model.zip")
    venv.save(str(out_dir / VECNORM))
    return out_dir / "model.zip"


def make_callbacks(args, tcfg, task_cfg, out: Path, venv):
    from stable_baselines3.common.callbacks import BaseCallback

    n_envs = int(tcfg["n_envs"])
    eval_freq = int(tcfg.get("eval_freq", 0) or 0)
    ckpt_freq = int(tcfg.get("checkpoint_freq", 0) or 0)
    n_eval = int(tcfg.get("eval_episodes", 50))
    seed0 = int(tcfg.get("eval_seed0", 100_000))
    eval_seeds = list(range(seed0, seed0 + n_eval))
    workers = int(tcfg.get("eval_workers", n_envs))

    class TrainCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.ep: list[dict] = []
            self.next_eval = eval_freq
            self.next_ckpt = ckpt_freq
            self.best = -math.inf
            self.t0 = time.perf_counter()
            self.eval_time = 0.0
            self.csv_path = out / "eval.csv"
            self.ref = None
            self.last_eval_ts = None

        def _on_training_start(self) -> None:
            ts0 = self.model.num_timesteps        # > 0 when warm-starting (--init-from)
            self.next_eval = ts0 + eval_freq
            self.next_ckpt = ts0 + ckpt_freq
            if eval_freq > 0 and tcfg.get("eval_baseline", True):
                t = time.perf_counter()
                self.ref = run_eval(task_cfg, None, eval_seeds, workers)
                self.eval_time += time.perf_counter() - t
                with open(out / "eval_baseline.json", "w") as f:
                    json.dump(self.ref, f, indent=2)
                print(f"[train] baseline on eval seeds: success {self.ref['success_rate']:.3f} "
                      f"pods {self.ref['impulse_pods']:.0f} jets {self.ref['impulse_jets']:.0f}",
                      flush=True)

        def _on_step(self) -> bool:
            for info, done in zip(self.locals["infos"], self.locals["dones"]):
                if done and "safe" in info:
                    self.ep.append({k: float(info[k]) for k in INFO_KEYS if k in info})
            ts = self.num_timesteps
            if ckpt_freq > 0 and ts >= self.next_ckpt:
                self.next_ckpt += ckpt_freq
                _save(self.model, venv, out / "checkpoints" / f"step_{ts}")
            if eval_freq > 0 and ts >= self.next_eval:
                self.next_eval += eval_freq
                self.evaluate(ts)
            return True

        def _on_rollout_end(self) -> None:
            if not self.ep:
                return
            for k in INFO_KEYS:
                v = [e[k] for e in self.ep if k in e and np.isfinite(e[k])]
                if v:
                    self.logger.record(f"rollout/{k}", float(np.mean(v)))
            self.logger.record("rollout/n_episodes", len(self.ep))
            std = getattr(self.model.policy, "log_std", None)
            if std is not None:
                self.logger.record("train/action_std", float(std.exp().mean().item()))
            self.ep = []

        def evaluate(self, ts: int) -> dict:
            self.last_eval_ts = ts
            t = time.perf_counter()
            d = _save(self.model, venv, out / "eval_tmp")
            st = run_eval(task_cfg, str(d), eval_seeds, workers)
            self.eval_time += time.perf_counter() - t
            score = eval_score(st, self.ref)
            row = {"timesteps": ts, "wall_s": round(time.perf_counter() - self.t0, 1),
                   "score": score, **{k: v for k, v in st.items() if k != "failures"},
                   "failures": json.dumps(st["failures"])}
            new = not self.csv_path.exists()
            with open(self.csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
            is_best = score > self.best
            if is_best:
                self.best = score
                _save(self.model, venv, out / "best")
                with open(out / "best" / "eval.json", "w") as f:
                    json.dump(row, f, indent=2)
            print(f"[train] eval @ {ts}: success {st['success_rate']:.3f} pods "
                  f"{st['impulse_pods']:.0f} jets {st['impulse_jets']:.0f} "
                  f"vz {st['impact_v_vertical']:.2f} vh {st['impact_v_horizontal']:.2f} "
                  f"ret {st['episode_return']:.2f} score {score:.2f}"
                  + ("  *best*" if is_best else ""), flush=True)
            return st

    return TrainCallback()


# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/train_ppo.yaml")
    ap.add_argument("--quick", action="store_true", help="laptop run: 4 envs, 200k steps")
    ap.add_argument("--total-timesteps", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None, help="run dir (default <out_dir>/ppo_<timestamp>)")
    ap.add_argument("--eval-freq", type=int, default=None, help="timesteps between evals (0=off)")
    ap.add_argument("--eval-episodes", type=int, default=None)
    ap.add_argument("--init-from", default=None,
                    help="warm-start from a model.zip (+ vecnormalize.pkl next to it); "
                         "--total-timesteps then counts additional steps")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="task config override, e.g. --set reward.k3_impulse=0.004")
    ap.add_argument("--train-set", action="append", default=[], metavar="KEY=VAL",
                    help="train config override, e.g. --train-set ppo.learning_rate=1e-4")
    ap.add_argument("--no-subproc", action="store_true", help="DummyVecEnv (tests/debug)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import torch
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import VecNormalize

    tcfg, task_cfg = build_configs(args)
    seed = int(tcfg.get("seed", 0))
    n_envs = int(tcfg["n_envs"])
    total = int(tcfg["total_timesteps"])
    out = Path(args.out or Path(tcfg.get("out_dir", "runs")) / f"ppo_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    save_run_metadata(out, {"train": tcfg, "task": task_cfg},
                      extra={"script": "scripts/train.py", "argv": sys.argv[1:] if argv is None else argv,
                             "status": "running"})

    from stable_baselines3.common.utils import set_random_seed
    set_random_seed(seed)
    venv = make_vec_env(task_cfg, n_envs, seed, subproc=not args.no_subproc)
    nc = tcfg.get("normalize", {}) or {}
    pc = dict(tcfg["ppo"])
    if args.init_from:
        from suitsim.control.residual_policy import vecnormalize_path
        vn = vecnormalize_path(args.init_from)
        venv = VecNormalize.load(str(vn), venv) if vn else VecNormalize(venv, norm_obs=True,
                                                                         norm_reward=False)
        venv.training = True
        model = PPO.load(args.init_from, env=venv, device="cpu",
                         custom_objects={"learning_rate": float(pc["learning_rate"])})
        model.set_random_seed(seed)
    else:
        venv = VecNormalize(venv, norm_obs=bool(nc.get("obs", True)),
                            norm_reward=bool(nc.get("reward", False)),
                            clip_obs=float(nc.get("clip_obs", 10.0)),
                            gamma=float(pc.get("gamma", 0.99)))
        hidden = list(pc.pop("policy_hidden", [128, 128]))
        log_std_init = float(pc.pop("log_std_init", -2.0))
        policy_kwargs = {"net_arch": {"pi": hidden, "vf": hidden}, "log_std_init": log_std_init,
                         "ortho_init": True}
        ppo_keys = ("learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda",
                    "clip_range", "ent_coef", "vf_coef", "max_grad_norm", "target_kl")
        kw = {k: pc[k] for k in ppo_keys if k in pc}
        for k in ("n_steps", "batch_size", "n_epochs"):
            if k in kw:
                kw[k] = int(kw[k])
        model = PPO("MlpPolicy", venv, policy_kwargs=policy_kwargs, seed=seed, device="cpu",
                    verbose=0, **kw)
        # policy starts as the baseline: zero residual mean (ortho gain 0.01 on the action
        # head is already ~0; zero it exactly so a=0 at init)
        with torch.no_grad():
            model.policy.action_net.weight.mul_(float(tcfg.get("action_init_scale", 0.0)))
            model.policy.action_net.bias.zero_()
    model.set_logger(configure(str(out), ["stdout", "csv"]))

    cb = make_callbacks(args, tcfg, task_cfg, out, venv)
    print(f"[train] {total} steps, {n_envs} envs, seed {seed} -> {out}", flush=True)
    t0 = time.perf_counter()
    model.learn(total_timesteps=total, callback=cb, progress_bar=False,
                reset_num_timesteps=not args.init_from)
    wall = time.perf_counter() - t0
    _save(model, venv, out)
    train_wall = max(wall - cb.eval_time, 1e-9)
    if (int(tcfg.get("eval_freq", 0) or 0) > 0 and tcfg.get("final_eval", True)
            and cb.last_eval_ts != model.num_timesteps):
        cb.evaluate(model.num_timesteps)
    venv.close()

    fps = total / train_wall
    timing = {"total_timesteps": total, "n_envs": n_envs, "wall_s": wall,
              "eval_wall_s": cb.eval_time, "train_wall_s": train_wall,
              "steps_per_s": fps, "s_per_100k_steps": 1e5 / fps,
              "projected_20M_hours_at_this_rate": 20e6 / fps / 3600.0,
              "projected_20M_hours_16_envs_linear_scaling":
                  20e6 / (fps * 16.0 / n_envs) / 3600.0}
    with open(out / "timing.json", "w") as f:
        json.dump(timing, f, indent=2)
    save_run_metadata(out, {"train": tcfg, "task": task_cfg},
                      extra={"script": "scripts/train.py",
                             "argv": sys.argv[1:] if argv is None else argv,
                             "status": "done", "timing": timing})
    print(f"[train] done in {wall / 60:.1f} min ({fps:.0f} steps/s, "
          f"{1e5 / fps / 60:.1f} min/100k) -> {out / 'model.zip'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
