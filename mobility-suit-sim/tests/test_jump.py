"""T2 boosted jump: env API, platform, determinism, nominal jump succeeds."""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from suitsim.control.jump import BaselineJumpController
from suitsim.envs import make_env
from suitsim.envs.boosted_jump import PLATFORM_GEOM, BoostedJumpEnv
from suitsim.interfaces import EPISODE_INFO_KEYS, load_config

NOMINAL = {"randomization": {"enabled": False},
           "task": {"fixed": {"yaw_deg": 0.0, "sway_deg": 0.0}}}


@pytest.fixture(scope="module")
def env_nom():
    env = make_env("boosted_jump", load_config("task_jump.yaml", NOMINAL))
    yield env
    env.close()


def run(env, seed):
    env.reset(seed=seed)
    n = 0
    while True:
        obs, r, term, trunc, info = env.step(np.zeros(6, np.float32))
        n += 1
        assert obs.shape == env.observation_space.shape and np.all(np.isfinite(obs))
        if term or trunc:
            return info, n


def test_registry_api_and_platform(env_nom):
    env = env_nom
    assert isinstance(env, BoostedJumpEnv)
    assert isinstance(env.controller, BaselineJumpController)
    assert hasattr(env.unwrapped, "thrusters")
    m = env.model
    pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, PLATFORM_GEOM)
    assert pid >= 0 and m.geom_bodyid[pid] == 0                    # static world geom
    assert m.geom_type[pid] == mujoco.mjtGeom.mjGEOM_BOX
    cfg = env.cfg
    H = cfg["task"]["platform_height"]
    obs, info0 = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert "platform_distance" in info0["episode_params"]
    mujoco.mj_forward(m, env.data)
    top = env.data.geom_xpos[pid][2] + m.geom_size[pid][2]
    assert top == pytest.approx(H)
    # collides with body geoms (contype 1 / conaffinity 2), not with the ground
    body_g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "geom_foot_l")
    assert (m.geom_contype[pid] & m.geom_conaffinity[body_g]) or \
        (m.geom_contype[body_g] & m.geom_conaffinity[pid])
    g = env.ground_id
    assert not ((m.geom_contype[pid] & m.geom_conaffinity[g])
                or (m.geom_contype[g] & m.geom_conaffinity[pid]))
    # standing start: soles on the ground, no platform contact
    soles = env.data.site_xpos[env.sole_sites][:, 2]
    assert np.all(np.abs(soles) < 0.01)


def test_randomised_distance_moves_platform():
    cfg = load_config("task_jump.yaml")
    env = make_env("boosted_jump", cfg, controller="none")
    ds = set()
    for s in range(3):
        _, info = env.reset(seed=s)
        d = info["episode_params"]["platform_distance"]
        assert abs(d - 1.5) <= cfg["task"]["platform_distance_range"] + 1e-9
        assert env.model.geom_pos[env.platform_id][0] == pytest.approx(d)
        ds.add(round(d, 6))
    assert len(ds) == 3


def test_nominal_jump_succeeds_and_deterministic(env_nom):
    info, n = run(env_nom, 0)
    assert all(k in info for k in EPISODE_INFO_KEYS)
    assert "episode_params" in info
    assert info["success"], info["failure_reasons"]
    assert info["first_contact"] in ("foot_l", "foot_r")
    assert info["landed_on_platform"] and info["stayed_on_platform"]
    assert info["impact_v_vertical"] < 2.0 and info["impact_v_horizontal"] < 1.0
    assert info["impulse_gas_pods"] <= 2500.0 and info["impulse_gas_jets"] <= 600.0
    info2, n2 = run(env_nom, 0)
    assert n2 == n
    for k in ("impact_v_vertical", "impact_v_horizontal", "impulse_gas_pods",
              "episode_time", "tilt_at_impact_deg"):
        assert info2[k] == info[k]


def test_no_controller_never_lifts_off():
    env = make_env("boosted_jump", load_config("task_jump.yaml", NOMINAL), controller="none")
    env.reset(seed=1)
    for _ in range(25):                      # 0.5 s standing
        *_, term, trunc, info = env.step(np.zeros(6))
        assert not term
    assert env.t_liftoff is None and env.plat_first is None
