"""Task environments (gymnasium) + registry.

    env = make_env("drop_landing", cfg, controller="baseline", log_trajectory=False)

Registered: ``drop_landing`` (T1).  ``boosted_jump`` (T2) and ``push_recovery`` (T3) are
reserved names; they will subclass :class:`suitsim.envs.base.SuitEnvBase`.
"""
from __future__ import annotations

from suitsim.envs.base import CONTROLLERS, SuitEnvBase
from suitsim.envs.drop_landing import DropLandingEnv
from suitsim.envs.boosted_jump import BoostedJumpEnv

_REGISTRY = {"drop_landing": DropLandingEnv, "boosted_jump": BoostedJumpEnv}
from suitsim.envs.push_recovery import PushRecoveryEnv  # noqa: E402  (T3)
_REGISTRY["push_recovery"] = PushRecoveryEnv
_RESERVED = ("boosted_jump", "push_recovery")


def register(name: str, cls) -> None:
    _REGISTRY[name] = cls


def make_env(task_name: str, cfg: dict, **kw):
    """Build the env for ``task_name`` (kwargs: controller, render_mode, log_trajectory,
    policy_dt)."""
    if task_name in _REGISTRY:
        return _REGISTRY[task_name](cfg, **kw)
    if task_name in _RESERVED:
        raise NotImplementedError(f"task '{task_name}' is reserved but not implemented yet")
    raise KeyError(f"unknown task '{task_name}'; known: {sorted(_REGISTRY)}")


__all__ = ["make_env", "register", "DropLandingEnv", "SuitEnvBase", "CONTROLLERS"]
