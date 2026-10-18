"""Residual policy (M5): zero residual == baseline; the SB3 wrapper loads and runs."""
from __future__ import annotations

import numpy as np
import pytest

from suitsim.envs import make_env
from suitsim.eval.batch import load_sb3_policy, results_equal, run_episode
from suitsim.interfaces import EPISODE_INFO_KEYS, load_config

# short drops keep the tests fast; randomisation stays on
SHORT = {"task": {"drop_height": [1.0, 1.6]}}


@pytest.fixture(scope="module")
def cfg():
    return load_config("task_drop.yaml", SHORT)


def test_zero_residual_equals_baseline(cfg):
    base = make_env("drop_landing", cfg, controller="baseline")
    res = make_env("drop_landing", cfg, controller="residual")
    for seed in (0, 7):
        a = run_episode(base, None, seed)
        b = run_episode(res, None, seed)            # policy None -> zero action
        assert a["complete_info"] and b["complete_info"]
        assert results_equal([a], [b], tol=0.0), seed
        assert a["steps"] == b["steps"] and a["episode_return"] == b["episode_return"]


def test_nonzero_residual_changes_episode(cfg):
    base = make_env("drop_landing", cfg, controller="baseline")
    res = make_env("drop_landing", cfg, controller="residual")
    a = run_episode(base, None, 0)
    b = run_episode(res, lambda o: np.array([0, 0, 0.5, 0, 0, 0], np.float32), 0)
    assert b["impulse_gas_pods"] > a["impulse_gas_pods"]


@pytest.fixture(scope="module")
def tiny_model(cfg, tmp_path_factory):
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    d = tmp_path_factory.mktemp("tiny_ppo")
    venv = VecNormalize(DummyVecEnv([lambda: make_env("drop_landing", cfg, controller="residual")]),
                        norm_obs=True, norm_reward=False)
    venv.seed(0)
    model = PPO("MlpPolicy", venv, n_steps=256, batch_size=128, n_epochs=2, seed=0,
                device="cpu", policy_kwargs={"log_std_init": -2.0, "net_arch": [32, 32]})
    model.learn(512)
    model.save(d / "model.zip")
    venv.save(str(d / "vecnormalize.pkl"))
    venv.close()
    return d / "model.zip"


def test_policy_wrapper_loads_and_runs(cfg, tiny_model):
    from suitsim.control.residual_policy import ResidualPolicy, make_policy_fn

    pol = ResidualPolicy(tiny_model)
    assert pol.vecnormalize_path is not None and pol.obs_mean is not None
    env = make_env("drop_landing", cfg, controller="residual")
    obs, _ = env.reset(seed=0)
    a = pol.act(obs)
    assert a.shape == (6,) and a.dtype == np.float32 and np.all(np.abs(a) <= 1.0)
    assert np.array_equal(a, pol.act(obs))                       # deterministic
    # same normalisation + action as the eval loader used by scripts/evaluate.py
    assert np.allclose(a, np.clip(load_sb3_policy(tiny_model)(obs), -1, 1), atol=1e-6)
    fn = make_policy_fn(tiny_model)
    r1 = run_episode(env, fn, 3)
    r2 = run_episode(env, fn, 3)
    assert r1["complete_info"] and results_equal([r1], [r2])
    assert all(k in r1 for k in EPISODE_INFO_KEYS)


def test_margin_reward_shaping(cfg):
    env = make_env("drop_landing", load_config("task_drop.yaml", {"reward": {"k_margin": 5.0}}),
                   controller="residual")
    env.reset(seed=0)
    info = {"impact_v_vertical": 1.5, "impact_v_horizontal": 0.3, "tilt_at_impact_deg": 5.0,
            "safe": True, "impulse_gas_pods": 0.0, "impulse_gas_jets": 0.0,
            "bad_contact": False, "first_contact": "foot_l"}
    r0 = env._reward(True, info)
    r1 = env._reward(True, dict(info, impact_v_vertical=1.85))        # 0.15 into the margin
    k1 = env.rw["k1_impact_speed2"]
    assert r0 - r1 == pytest.approx(5.0 * 0.25 + k1 * (1.85 ** 2 - 1.5 ** 2))
    env.rw["k_margin"] = 0.0
    assert env._reward(True, info) == pytest.approx(r0)                # inactive below margin
