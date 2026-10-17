"""Baseline landing controller: unit behaviour + M4 criterion (fixed 3 m upright drop)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from suitsim.control.baseline import BaselineLandingController
from suitsim.envs import make_env
from suitsim.interfaces import BUDGET_KEYS, FEET, load_config

FIXED_3M = {"task": {"fixed": {"drop_height": 3.0, "tilt_deg": 0.0, "angvel": 0.0}},
            "randomization": {"enabled": False}}


@pytest.fixture(scope="module")
def cfg():
    return load_config("task_drop.yaml")


def est(h=3.0, vz=0.0, ff=True, quat=(1, 0, 0, 0), w=(0, 0, 0), contact=(False, False), vxy=(0, 0)):
    q = np.asarray(quat, float)
    q /= np.linalg.norm(q)
    tilt = 2 * math.acos(min(1.0, abs(q[0]))) if abs(q[3]) < 1e-12 else 0.0
    return {"t": 0.0, "com_height": h + 1.0, "feet_height": h,
            "com_vel": np.array([vxy[0], vxy[1], vz]), "torso_quat": q,
            "torso_angvel": np.asarray(w, float), "tilt": tilt,
            "joint_pos": np.zeros(22), "joint_vel": np.zeros(22),
            "budgets": {k: 1.0 for k in BUDGET_KEYS}, "free_fall": ff,
            "time_since_fall": 0.1 if ff else 0.0, "foot_contact": np.asarray(contact)}


def test_idle_before_fall(cfg):
    c = BaselineLandingController(cfg)
    out = c.compute(est(ff=False, vz=0.0))
    assert out.phase == "idle" and not np.any(out.wrench)


def test_fall_fallback_on_descent(cfg):
    c = BaselineLandingController(cfg)
    out = c.compute(est(ff=False, vz=-1.0))
    assert out.phase != "idle"


def test_profile_and_braking(cfg):
    c = BaselineLandingController(cfg)
    b = cfg["baseline"]
    vr, a = c.v_ref(b["target_height"])
    assert vr == pytest.approx(-b["target_speed"]) and a == 0.0
    hs = np.linspace(0.3, 5, 20)
    v = [c.v_ref(h)[0] for h in hs]
    assert np.all(np.diff(v) < 0)                      # faster allowed higher up
    # slower than the profile -> no braking thrust; faster -> strong thrust
    out = c.compute(est(h=4.0, vz=-1.0))
    assert out.wrench[2] == pytest.approx(0.0, abs=1.0) and out.phase == "attitude"
    out = c.compute(est(h=1.0, vz=-6.0))
    assert out.phase == "brake" and out.wrench[2] > 1.5 * 9.81 * b["mass_nominal"]


def test_attitude_restoring_torque(cfg):
    c = BaselineLandingController(cfg)
    ang = math.radians(20)
    q = (math.cos(ang / 2), math.sin(ang / 2), 0, 0)    # rolled +20 deg about x
    out = c.compute(est(h=4.0, vz=-0.8, quat=q))
    assert out.wrench[3] < 0                            # negative roll torque restores
    q = (math.cos(ang / 2), 0, math.sin(ang / 2), 0)    # pitched +20 deg about y
    out = BaselineLandingController(cfg).compute(est(h=4.0, vz=-0.8, quat=q))
    assert out.wrench[4] < 0
    out = BaselineLandingController(cfg).compute(est(h=4.0, vz=-0.8, w=(0, 0, 2.0)))
    assert out.wrench[5] < 0                            # yaw rate damped


def test_touchdown_cuts_thrust(cfg):
    c = BaselineLandingController(cfg)
    c.compute(est(h=1.0, vz=-3.0))
    out = c.compute(est(h=0.0, vz=-1.5, contact=(True, False)))
    assert out.phase == "landed" and not np.any(out.wrench)
    out = c.compute(est(h=0.5, vz=-1.0, contact=(False, False)))
    assert out.phase == "landed"                        # latched


def test_airbags_on_fast_approach(cfg):
    c = BaselineLandingController(cfg)
    out = c.compute(est(h=0.3, vz=-5.0))
    assert out.deploy_airbags
    c = BaselineLandingController(cfg)
    out = c.compute(est(h=0.3, vz=-1.5))
    assert not out.deploy_airbags


def test_m4_fixed_3m_upright_safe():
    """Milestone M4, first criterion: baseline lands a fixed 3 m upright drop safely."""
    cfg = load_config("task_drop.yaml", FIXED_3M)
    env = make_env("drop_landing", cfg, controller="baseline")
    env.reset(seed=0)
    while True:
        _, _, term, trunc, info = env.step(np.zeros(6, np.float32))
        if term or trunc:
            break
    assert term and not trunc
    assert info["safe"], info
    assert info["first_contact"] in FEET
    assert info["impact_v_vertical"] < cfg["safe_landing"]["max_vertical_speed"]
    assert info["impact_v_horizontal"] < cfg["safe_landing"]["max_horizontal_speed"]
    assert info["tilt_at_impact_deg"] < cfg["safe_landing"]["max_tilt_deg"]
    assert info["impulse_gas_pods"] < cfg["suit"]["budgets"]["impulse_gas_pods"]
    assert info["impulse_gas_jets"] < cfg["suit"]["budgets"]["impulse_gas_jets"]
