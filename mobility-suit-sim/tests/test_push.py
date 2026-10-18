"""T3 push recovery on a beam: env API, determinism, beam, controller vs no controller."""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from suitsim.envs import make_env
from suitsim.envs.push_recovery import BEAM_GEOM, PushRecoveryEnv
from suitsim.interfaces import EPISODE_INFO_KEYS, load_config

NO_DR = {"randomization": {"enabled": False}}


def _cfg(**fixed):
    ov = dict(NO_DR)
    ov["task"] = {"fixed": fixed or None}
    return load_config("task_push.yaml", ov)


def run(env, seed):
    env.reset(seed=seed)
    while True:
        obs, r, term, trunc, info = env.step(None)
        assert obs.shape == env.observation_space.shape and np.all(np.isfinite(obs))
        if term or trunc:
            return info


@pytest.fixture(scope="module")
def env_lateral():
    return make_env("push_recovery", _cfg(push_direction_deg=90.0, push_time=0.6),
                    controller="baseline")


def test_registry_api_and_beam():
    cfg = load_config("task_push.yaml")
    env = make_env("push_recovery", cfg, controller="none")
    assert isinstance(env, PushRecoveryEnv)
    assert hasattr(env.unwrapped, "thrusters")
    obs, info = env.reset(seed=0)
    p = info["episode_params"]
    for k in ("push_direction_deg", "push_time", "push_force", "mass_scale", "friction"):
        assert k in p
    assert 0.5 <= p["push_time"] <= 1.5 and 0.0 <= p["push_direction_deg"] < 360.0
    m, d = env.model, env.data
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, BEAM_GEOM)
    assert g >= 0 and m.geom_bodyid[g] == 0
    assert m.geom_size[g][1] == pytest.approx(0.15)
    assert d.geom_xpos[g][2] + m.geom_size[g][2] == pytest.approx(1.0)
    soles = d.site_xpos[env.sole_sites][:, 2]
    assert np.allclose(soles, 1.0, atol=0.02)          # standing on the beam top
    assert m.geom_friction[g][0] == pytest.approx(p["friction"])
    env.step(env.action_space.sample())
    assert env.data.time == pytest.approx(env.policy_dt)


def test_no_push_stays_on_beam():
    env = make_env("push_recovery", _cfg(push_force=0.0), controller="none")
    info = run(env, 0)
    assert all(k in info for k in EPISODE_INFO_KEYS)
    assert info["success"] and info["safe"] and info["first_contact"] in ("foot_l", "foot_r")
    assert info["episode_time"] == pytest.approx(4.0, abs=0.03)
    assert np.isfinite(info["impact_v_vertical"]) and np.isfinite(info["tilt_at_impact_deg"])


def test_lateral_push_falls_without_controller():
    env = make_env("push_recovery", _cfg(push_direction_deg=90.0, push_time=0.6),
                   controller="none")
    info = run(env, 0)
    assert not info["success"]
    assert info["episode_time"] < 4.0                   # fall ends the episode early


def test_lateral_push_recovered_and_deterministic(env_lateral):
    a = run(env_lateral, 3)
    assert a["success"], a["failure_reasons"]
    assert a["impulse_gas_jets"] > 0.0
    b = run(env_lateral, 3)
    for k in ("impulse_gas_jets", "impulse_gas_pods", "final_tilt_deg", "final_com_offset_y"):
        assert a[k] == pytest.approx(b[k], rel=0, abs=1e-12)
