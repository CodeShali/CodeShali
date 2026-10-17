"""Env API contract tests for T1 (DropLandingEnv)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from suitsim.envs import DropLandingEnv, make_env
from suitsim.interfaces import BUDGET_KEYS, EPISODE_INFO_KEYS, THRUSTER_CLASSES, load_config

FIXED_3M = {"task": {"fixed": {"drop_height": 3.0, "tilt_deg": 0.0, "angvel": 0.0}},
            "randomization": {"enabled": False}}
PARAM_KEYS = ("drop_height", "tilt_deg", "angvel", "mass_scale", "friction", "noise_scale",
              "fmax_scale_mean", "tau_scale_mean")


@pytest.fixture(scope="module")
def cfg():
    return load_config("task_drop.yaml")


@pytest.fixture(scope="module")
def cfg3():
    return load_config("task_drop.yaml", FIXED_3M)


def run(env, seed, action=None):
    obs, info0 = env.reset(seed=seed)
    a = np.zeros(6, np.float32) if action is None else action
    n, total = 0, 0.0
    while True:
        obs, r, term, trunc, info = env.step(a)
        assert env.observation_space.contains(obs) or obs.shape == env.observation_space.shape
        total += r
        n += 1
        if term or trunc:
            return obs, info, total, n


def test_registry_and_spaces(cfg):
    env = make_env("drop_landing", cfg, controller="none")
    assert isinstance(env, DropLandingEnv)
    assert env.action_space.shape == (6,)
    obs, info = env.reset(seed=0)
    assert obs.dtype == np.float32 and obs.shape == env.observation_space.shape
    assert np.all(np.isfinite(obs))
    assert set(PARAM_KEYS) <= set(info["episode_params"])
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert obs.shape == env.observation_space.shape and isinstance(r, float)
    # one env step = policy_dt
    assert env.unwrapped.data.time == pytest.approx(env.policy_dt, abs=1e-9)
    assert env.n_ctrl == 10 and env.n_phys == 2
    assert env.unwrapped.model is not None and env.unwrapped.data is not None
    assert hasattr(env.unwrapped, "thrusters")
    with pytest.raises(NotImplementedError):
        make_env("boosted_jump", cfg)
    with pytest.raises(KeyError):
        make_env("nope", cfg)
    with pytest.raises(ValueError):
        make_env("drop_landing", cfg, controller="pid")


def test_policy_dt_kwarg(cfg):
    env = make_env("drop_landing", cfg, controller="none", policy_dt=0.01)
    env.reset(seed=0)
    env.step(np.zeros(6))
    assert env.data.time == pytest.approx(0.01, abs=1e-9)


def test_none_controller_unsafe_and_info(cfg3):
    env = make_env("drop_landing", cfg3, controller="none", log_trajectory=True)
    _, info, ret, _ = run(env, 0)
    assert set(EPISODE_INFO_KEYS) <= set(info)
    assert set(PARAM_KEYS) <= set(info["episode_params"])
    assert info["episode_params"]["drop_height"] == 3.0
    assert not info["safe"] and not info["success"]
    assert info["first_contact"] in ("foot_l", "foot_r")
    assert info["impact_v_vertical"] == pytest.approx(math.sqrt(2 * 9.81 * 3.0), rel=0.05)
    assert info["impulse_gas_pods"] == 0.0 and info["impulse_gas_jets"] == 0.0
    assert not info["airbags_deployed"]
    tr = info["trajectory"]
    assert len(tr) > 100
    keys = {"t", "com_height", "com_vel", "tilt", "forces_by_class", "budgets", "phase",
            "est_com_height", "est_com_vel", "est_tilt"}
    assert keys <= set(tr[0])
    assert set(THRUSTER_CLASSES) <= set(tr[0]["forces_by_class"])
    assert set(BUDGET_KEYS) <= set(tr[0]["budgets"])
    assert ret < 0


def test_determinism_and_budgets(cfg):
    a = make_env("drop_landing", cfg, controller="baseline", log_trajectory=True)
    b = make_env("drop_landing", cfg, controller="baseline")
    seed = 3
    _, ia, ra, na = run(a, seed)
    _, ib, rb, nb = run(b, seed)
    tr = ia.pop("trajectory")
    assert na == nb and ra == rb
    for k in EPISODE_INFO_KEYS:
        assert ia[k] == ib[k] or (isinstance(ia[k], float) and math.isnan(ia[k]) and math.isnan(ib[k]))
    assert ia["episode_params"] == ib["episode_params"]
    caps = cfg["suit"]["budgets"]
    for p in tr:
        for k in BUDGET_KEYS:
            assert 0.0 <= p["budgets"][k] <= caps[k]
    assert 0 < ia["impulse_gas_pods"] <= caps["impulse_gas_pods"]
    # a different seed gives a different episode
    _, ic, _, _ = run(b, seed + 1)
    assert ic["episode_params"] != ia["episode_params"]


def test_residual_zero_action_matches_baseline(cfg3):
    base = make_env("drop_landing", cfg3, controller="baseline")
    res = make_env("drop_landing", cfg3, controller="residual")
    _, ib, _, _ = run(base, 1)
    _, ir, _, _ = run(res, 1)
    for k in EPISODE_INFO_KEYS:
        assert ib[k] == ir[k]
    # a constant upward residual changes the outcome
    _, iu, _, _ = run(res, 1, np.array([0, 0, 1, 0, 0, 0], np.float32))
    assert iu["impulse_gas_pods"] != ib["impulse_gas_pods"]


_RENDER_SCRIPT = """
import numpy as np
from suitsim.envs import make_env
from suitsim.interfaces import load_config
cfg = load_config("task_drop.yaml", {"task": {"fixed": {"drop_height": 3.0, "tilt_deg": 0.0,
                                               "angvel": 0.0}}})
env = make_env("drop_landing", cfg, controller="none", render_mode="rgb_array",
               render_size=(60, 80))
env.reset(seed=0)
f = env.render()
assert f.shape == (60, 80, 3) and f.dtype == np.uint8 and f.std() > 0, f.shape
env.close()
print("RENDER_OK")
"""


def test_render():
    """Offscreen render via the 'track' camera. Run in a fresh interpreter: the GL backend is
    fixed when mujoco is first imported, which other tests in this process already did."""
    import os
    import subprocess
    import sys
    env = {k: v for k, v in os.environ.items() if k != "MUJOCO_GL"}
    out = subprocess.run([sys.executable, "-c", _RENDER_SCRIPT], capture_output=True,
                         text=True, timeout=120, env=env)
    if "RENDER_OK" not in out.stdout:
        if "osmesa" in out.stderr.lower() or "gl" in out.stderr.lower():
            pytest.skip(f"no offscreen GL: {out.stderr[-300:]}")
        raise AssertionError(out.stderr[-2000:])


def test_render_guard_when_gl_locked(cfg3, monkeypatch):
    import suitsim.envs.base as base
    monkeypatch.setattr(base, "_GL_LOCKED_OUT", True)
    env = make_env("drop_landing", cfg3, controller="none", render_mode="rgb_array")
    env.reset(seed=0)
    with pytest.raises(RuntimeError):
        env.render()
