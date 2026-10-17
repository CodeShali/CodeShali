"""Batch episode runner (optionally multi-process, spawn-safe).

Envs and policies are NOT pickled across processes. Instead an :class:`EnvSpec`
(config path + overrides + controller + policy path + env kwargs) is sent to each worker,
which builds its own env / policy once and then runs its share of the seeds.

    spec = EnvSpec("configs/task_drop.yaml", controller="baseline")
    results = run_episodes(spec, None, seeds=range(1000), n_workers=4)
"""
from __future__ import annotations

import importlib
import math
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from suitsim.interfaces import EPISODE_INFO_KEYS, PROJECT_ROOT, load_config

PolicyFn = Callable[[np.ndarray], np.ndarray]


# --------------------------------------------------------------------------------------
# Env / policy specification (picklable)
# --------------------------------------------------------------------------------------
def import_object(path: str) -> Any:
    """Import ``"package.module:attr"`` (or ``"package.module.attr"``)."""
    if ":" in path:
        mod, attr = path.split(":", 1)
    else:
        mod, attr = path.rsplit(".", 1)
    obj = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def parse_overrides(items: list[str]) -> dict:
    """``--set a.b=3 --set c=[1,2]`` -> nested dict (values parsed as YAML)."""
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


@dataclass
class EnvSpec:
    """Everything needed to (re)build an env + policy inside any process."""
    task_cfg: str | Path | None = None          # e.g. configs/task_drop.yaml
    overrides: dict = field(default_factory=dict)
    controller: str = "baseline"                # baseline | residual | none
    policy_path: str | Path | None = None       # SB3 .zip for 'residual'
    env_kwargs: dict = field(default_factory=dict)
    factory: str | None = None                  # "module:callable"(cfg, controller=, **kw)
    cfg: dict | None = None                     # pre-loaded config (wins over task_cfg)

    def config(self) -> dict:
        if self.cfg is not None:
            return self.cfg
        if self.task_cfg is None:
            return dict(self.overrides or {})
        path = Path(self.task_cfg)
        if not path.is_absolute() and not path.exists() and (PROJECT_ROOT / path).exists():
            path = PROJECT_ROOT / path        # allow 'configs/x.yaml' from any cwd
        return load_config(path, self.overrides)

    def make_env(self, **extra_kwargs):
        cfg = self.config()
        kw = {**self.env_kwargs, **extra_kwargs}
        if self.factory:
            return import_object(self.factory)(cfg, controller=self.controller, **kw)
        from suitsim.envs import make_env  # lazy: envs are built by another workstream
        return make_env(cfg["task"]["name"], cfg, controller=self.controller, **kw)

    def make_policy(self) -> PolicyFn | None:
        if self.policy_path is None or self.controller != "residual":
            return None
        return load_sb3_policy(self.policy_path)

    def __call__(self, **kw):
        return self.make_env(**kw)


def load_sb3_policy(path: str | Path, deterministic: bool = True) -> PolicyFn:
    """Load an SB3 model (PPO/SAC/TD3 tried in order) as ``obs -> action``.

    If ``vecnormalize.pkl`` (or ``<stem>_vecnormalize.pkl``) sits next to the model, its
    observation normalisation is applied.
    """
    import pickle

    path = Path(path)
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    model = None
    errors = []
    import stable_baselines3 as sb3
    for algo in ("PPO", "SAC", "TD3", "A2C"):
        try:
            model = getattr(sb3, algo).load(str(path), device="cpu")
            break
        except Exception as e:  # wrong algo class -> try next
            errors.append(f"{algo}: {e}")
    if model is None:
        raise RuntimeError(f"could not load policy {path}: {errors}")

    norm = None
    for cand in (path.parent / "vecnormalize.pkl", path.with_name(path.stem + "_vecnormalize.pkl")):
        if cand.exists():
            with open(cand, "rb") as f:
                vn = pickle.load(f)
            if getattr(vn, "norm_obs", True) and getattr(vn, "obs_rms", None) is not None:
                norm = (np.asarray(vn.obs_rms.mean), np.asarray(vn.obs_rms.var),
                        float(vn.clip_obs), float(vn.epsilon))
            break

    def policy(obs: np.ndarray) -> np.ndarray:
        o = np.asarray(obs, dtype=np.float32)
        if norm is not None:
            mean, var, clip, eps = norm
            o = np.clip((o - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32)
        act, _ = model.predict(o, deterministic=deterministic)
        return act

    return policy


# --------------------------------------------------------------------------------------
# Episode loop
# --------------------------------------------------------------------------------------
def _to_py(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _to_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_py(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def run_episode(env, policy_fn: PolicyFn | None, seed: int, max_steps: int = 100_000,
                keep_trajectory: bool = False, step_callback: Callable | None = None,
                reset_callback: Callable | None = None) -> dict:
    """Run one episode with ``env.reset(seed=seed)``; returns the final info + bookkeeping."""
    t0 = time.perf_counter()
    obs, _ = env.reset(seed=int(seed))
    if reset_callback is not None:
        reset_callback(env, obs)
    shape = env.action_space.shape
    zeros = np.zeros(shape, dtype=np.float32)
    ret, steps, info = 0.0, 0, {}
    terminated = truncated = False
    while steps < max_steps:
        action = zeros if policy_fn is None else np.asarray(policy_fn(obs), dtype=np.float32)
        obs, r, terminated, truncated, info = env.step(action)
        ret += float(r)
        steps += 1
        if step_callback is not None:
            step_callback(env, obs, info)
        if terminated or truncated:
            break
    info = dict(info or {})
    traj = info.pop("trajectory", None)
    rec = {k: _to_py(v) for k, v in info.items() if k != "terminal_observation"}
    rec.update(seed=int(seed), episode_return=ret, steps=steps,
               terminated=bool(terminated), truncated=bool(truncated),
               wall_time=time.perf_counter() - t0)
    rec["complete_info"] = all(k in info for k in EPISODE_INFO_KEYS)
    if keep_trajectory and traj is not None:
        rec["trajectory"] = _to_py(traj)
    return rec


# Worker state (one env + policy per process)
_W: dict[str, Any] = {}


def _worker_init(spec: EnvSpec, max_steps: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    _W["env"] = spec.make_env()
    _W["policy"] = spec.make_policy()
    _W["max_steps"] = max_steps


def _worker_run(seed: int) -> dict:
    try:
        return run_episode(_W["env"], _W["policy"], seed, _W["max_steps"])
    except Exception as e:  # keep the batch alive; record the crash
        return {"seed": int(seed), "error": f"{type(e).__name__}: {e}", "success": False,
                "safe": False, "complete_info": False}


def run_episodes(env_factory: EnvSpec | Callable[[], Any],
                 policy_fn: PolicyFn | None,
                 seeds: Iterable[int],
                 n_workers: int = 1,
                 max_steps: int = 100_000,
                 progress: bool = False) -> list[dict]:
    """Run one episode per seed, return per-episode result dicts in ``seeds`` order.

    ``env_factory``: an :class:`EnvSpec` (required for ``n_workers > 1``) or any
    zero-arg callable returning an env. ``policy_fn``: ``obs -> action`` or None (zeros);
    when None and ``env_factory`` is an EnvSpec with a ``policy_path``, the policy is
    loaded from it (inside each worker).
    """
    seeds = [int(s) for s in seeds]
    if not seeds:
        return []
    n_workers = max(1, min(int(n_workers), len(seeds)))
    if n_workers == 1:
        env = env_factory()
        if policy_fn is None and isinstance(env_factory, EnvSpec):
            policy_fn = env_factory.make_policy()
        out = []
        try:
            for i, s in enumerate(seeds):
                out.append(run_episode(env, policy_fn, s, max_steps))
                if progress and (i + 1) % max(1, len(seeds) // 20) == 0:
                    print(f"  {i + 1}/{len(seeds)} episodes", flush=True)
        finally:
            if hasattr(env, "close"):
                env.close()
        return out

    if not isinstance(env_factory, EnvSpec):
        raise TypeError("n_workers > 1 requires an EnvSpec env_factory (picklable)")
    if policy_fn is not None and env_factory.policy_path is None:
        raise TypeError("n_workers > 1: pass the policy via EnvSpec.policy_path")
    ctx = mp.get_context("spawn")
    chunk = max(1, math.ceil(len(seeds) / (n_workers * 8)))
    out: list[dict] = []
    with ctx.Pool(n_workers, initializer=_worker_init, initargs=(env_factory, max_steps)) as pool:
        for i, r in enumerate(pool.imap(_worker_run, seeds, chunksize=chunk)):
            out.append(r)
            if progress and (i + 1) % max(1, len(seeds) // 20) == 0:
                print(f"  {i + 1}/{len(seeds)} episodes", flush=True)
    return out


def results_equal(a: Sequence[dict], b: Sequence[dict],
                  keys: Sequence[str] = tuple(EPISODE_INFO_KEYS), tol: float = 1e-9) -> bool:
    """Compare two result lists on the ground-truth keys (ignores wall time)."""
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if ra.get("seed") != rb.get("seed"):
            return False
        for k in keys:
            va, vb = ra.get(k), rb.get(k)
            if isinstance(va, float) or isinstance(vb, float):
                if va is None or vb is None or abs(float(va) - float(vb)) > tol:
                    return False
            elif va != vb:
                return False
    return True
