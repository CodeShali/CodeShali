"""Residual RL policy wrapper (M5).

A residual policy outputs ``a in [-1, 1]^6``; the env scales it by
``residual.max_force / max_torque`` and adds it to the baseline wrench while the baseline is
active (see :meth:`suitsim.envs.base.SuitEnvBase._control_step`).  ``a = 0`` reproduces the
baseline exactly.

Artifacts written by ``scripts/train.py`` (and expected here)::

    runs/ppo_xxx/model.zip           SB3 PPO model
    runs/ppo_xxx/vecnormalize.pkl    VecNormalize statistics (observation normalisation)

Usage::

    pol = ResidualPolicy("runs/ppo_xxx/model.zip")
    a = pol.act(obs)                         # deterministic
    fn = make_policy_fn("runs/ppo_xxx/model.zip")   # obs -> action (eval/batch.py PolicyFn)

``scripts/evaluate.py --controller residual --policy runs/ppo_xxx/model.zip`` loads the same
artifacts through :func:`suitsim.eval.batch.load_sb3_policy` (identical normalisation).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable

import numpy as np

VECNORM_NAME = "vecnormalize.pkl"


def vecnormalize_path(model_path: str | Path) -> Path | None:
    """``vecnormalize.pkl`` next to the model, else ``<stem>_vecnormalize.pkl``, else None."""
    p = Path(model_path)
    for cand in (p.parent / VECNORM_NAME, p.with_name(p.stem + "_vecnormalize.pkl")):
        if cand.exists():
            return cand
    return None


class ResidualPolicy:
    """Deterministic ``obs -> action`` from an SB3 model (+ optional VecNormalize stats)."""

    def __init__(self, model_path: str | Path, vecnormalize: str | Path | None = None,
                 deterministic: bool = True, device: str = "cpu"):
        try:
            import torch
            torch.set_num_threads(1)
        except Exception:  # pragma: no cover
            pass
        from stable_baselines3 import PPO

        self.path = Path(model_path)
        if not self.path.exists() and self.path.with_suffix(".zip").exists():
            self.path = self.path.with_suffix(".zip")
        self.model = PPO.load(str(self.path), device=device)
        self.deterministic = bool(deterministic)
        vn_path = Path(vecnormalize) if vecnormalize else vecnormalize_path(self.path)
        self.obs_mean = self.obs_var = None
        self.clip_obs, self.epsilon = 10.0, 1e-8
        if vn_path is not None:
            with open(vn_path, "rb") as f:
                vn = pickle.load(f)
            if getattr(vn, "norm_obs", True) and getattr(vn, "obs_rms", None) is not None:
                self.obs_mean = np.asarray(vn.obs_rms.mean, np.float64)
                self.obs_var = np.asarray(vn.obs_rms.var, np.float64)
                self.clip_obs = float(vn.clip_obs)
                self.epsilon = float(vn.epsilon)
        self.vecnormalize_path = vn_path
        self.action_dim = int(np.prod(self.model.action_space.shape))

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        o = np.asarray(obs, np.float64)
        if self.obs_mean is None:
            return o.astype(np.float32)
        o = (o - self.obs_mean) / np.sqrt(self.obs_var + self.epsilon)
        return np.clip(o, -self.clip_obs, self.clip_obs).astype(np.float32)

    def act(self, obs: np.ndarray) -> np.ndarray:
        a, _ = self.model.predict(self.normalize(obs), deterministic=self.deterministic)
        return np.clip(np.asarray(a, np.float32), -1.0, 1.0)

    __call__ = act

    def reset(self) -> None:  # stateless MLP policy; kept for controller-like symmetry
        pass


def make_policy_fn(path: str | Path, deterministic: bool = True) -> Callable[[np.ndarray], np.ndarray]:
    """``obs -> action`` callable (the ``PolicyFn`` convention of suitsim.eval.batch)."""
    return ResidualPolicy(path, deterministic=deterministic).act


__all__ = ["ResidualPolicy", "make_policy_fn", "vecnormalize_path", "VECNORM_NAME"]
