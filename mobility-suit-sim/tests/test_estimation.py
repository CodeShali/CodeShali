"""State-estimator tests (A3) on A1's model."""
from __future__ import annotations

import math
import time

import mujoco
import numpy as np
import pytest

from suitsim.interfaces import ESTIMATE_KEYS, FEET, TORSO_BODY, foot_sole_site, load_config

model_pkg = pytest.importorskip("suitsim.model")
from suitsim.estimation import StateEstimator  # noqa: E402
from suitsim.sensors import SensorSuite  # noqa: E402

CFG = load_config()


@pytest.fixture(scope="module")
def model():
    return model_pkg.build_model(CFG)


def truth(m, d):
    """Ground-truth (test-side only) CoM height/velocity, torso quat, lowest sole height."""
    mujoco.mj_forward(m, d)            # derived quantities at the post-step state
    c, v = model_pkg.com_state(m, d)
    tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY)
    soles = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, foot_sole_site(f)) for f in FEET]
    return c, v, d.xquat[tid].copy(), float(d.site_xpos[soles, 2].min())


def quat_err_deg(qa, qb):
    dq = np.empty(4)
    mujoco.mju_mulQuat(dq, qa, np.array([qb[0], -qb[1], -qb[2], -qb[3]]))
    return math.degrees(2 * math.acos(min(1.0, abs(dq[0]))))


def tilt_err_deg(qa, qb):
    """Angle between the torso z axes (yaw-free attitude error)."""
    Ra, Rb = np.empty(9), np.empty(9)
    mujoco.mju_quat2Mat(Ra, qa)
    mujoco.mju_quat2Mat(Rb, qb)
    return math.degrees(math.acos(min(1.0, Ra.reshape(3, 3)[:, 2] @ Rb.reshape(3, 3)[:, 2])))


def run(m, cfg, setup, T, seed, noise_scale=1.0, apply_errors=True):
    d = mujoco.MjData(m)
    setup(m, d)
    rng = np.random.default_rng(seed)
    sens = SensorSuite(m, cfg, rng)
    sens.reset(m, d, noise_scale=noise_scale)
    est = StateEstimator(m, cfg)
    c, v, q, _ = truth(m, d)
    est.reset({"torso_quat": q, "com_vel": v, "com_height": c[2]}, rng,
              **({} if apply_errors else {"apply_errors": False}))
    nsub = int(round(cfg["sim"]["control_dt"] / m.opt.timestep))
    log = []
    for i in range(int(round(T / m.opt.timestep))):
        mujoco.mj_step(m, d)
        sens.record(m, d, None)
        if (i + 1) % nsub == 0:
            e = est.update(sens.read(), cfg["sim"]["control_dt"])
            c, v, q, fz = truth(m, d)
            log.append(dict(t=d.time, e=e, att=quat_err_deg(e["torso_quat"], q),
                            tilt=tilt_err_deg(e["torso_quat"], q),
                            vz=abs(e["com_vel"][2] - v[2]), v=np.linalg.norm(e["com_vel"] - v),
                            h=abs(e["com_height"] - c[2]), fh=abs(e["feet_height"] - fz)))
    return log, est, d


def fall_setup(seed, tilt_deg=20.0, angvel=1.0, height=3.0):
    r = np.random.default_rng(1000 + seed)
    ax = r.standard_normal(2)
    ax /= np.linalg.norm(ax)
    a = math.radians(tilt_deg)
    q = np.array([math.cos(a / 2), *(math.sin(a / 2) * ax), 0.0])
    w = r.standard_normal(3)
    w *= angvel / np.linalg.norm(w)

    def setup(m, d):
        model_pkg.set_initial_pose(m, d, height, q, w)
    return setup


def stand_setup(m, d):
    k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, model_pkg.STAND_KEY)
    mujoco.mj_resetDataKeyframe(m, d, k)
    mujoco.mj_forward(m, d)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_free_fall_accuracy(model, seed):
    log, est, _ = run(model, CFG, fall_setup(seed), 0.7, seed)
    att = max(x["att"] for x in log)
    vz = max(x["vz"] for x in log)
    h = max(x["h"] for x in log)
    assert att < 3.0, att
    assert vz < 0.3, vz
    assert h < 0.15, h
    assert max(x["fh"] for x in log) < 0.2
    det = next(x["t"] for x in log if x["e"]["free_fall"])
    assert 0.050 <= det <= 0.080, det
    last = log[-1]["e"]
    assert last["free_fall"] and last["time_since_fall"] > 0.6
    assert all(x["e"]["time_since_fall"] == 0.0 for x in log if not x["e"]["free_fall"])
    assert not last["foot_contact"].any()


def test_free_fall_noise_free_is_tight(model):
    log, _, _ = run(model, CFG, fall_setup(0), 0.7, 0, noise_scale=0.0, apply_errors=False)
    assert max(x["att"] for x in log) < 0.2
    assert max(x["vz"] for x in log) < 0.05
    assert max(x["h"] for x in log) < 0.03


def test_standing_no_free_fall(model):
    log, est, _ = run(model, CFG, stand_setup, 1.5, 0)
    assert not any(x["e"]["free_fall"] for x in log)
    assert all(x["e"]["time_since_fall"] == 0.0 for x in log)
    assert log[-1]["e"]["foot_contact"].all()
    assert max(x["h"] for x in log[25:]) < 0.05
    assert max(x["fh"] for x in log[25:]) < 0.05
    assert max(x["vz"] for x in log) < 0.15
    assert max(x["att"] for x in log) < 3.0
    # residual = slow postural sway acceleration read as tilt (a_h / g ~ 0.1 / 9.81 rad)
    assert log[-1]["tilt"] < 1.0
    # ... and the zero-velocity update keeps the horizontal drift bounded
    assert log[-1]["v"] < 0.1


def test_standing_tilt_correction(model):
    """A pure 3 deg tilt error is pulled out by the quasi-static gravity correction
    (yaw is unobservable from gravity and is not corrected)."""
    d = mujoco.MjData(model)
    stand_setup(model, d)
    rng = np.random.default_rng(0)
    sens = SensorSuite(model, CFG, rng)
    sens.reset(model, d)
    est = StateEstimator(model, CFG)
    c, v, q, _ = truth(model, d)
    dq = np.empty(4)
    mujoco.mju_axisAngle2Quat(dq, np.array([1.0, 0.0, 0.0]), math.radians(3.0))
    q_bad = np.empty(4)
    mujoco.mju_mulQuat(q_bad, dq, q)
    est.reset({"torso_quat": q_bad, "com_vel": v, "com_height": c[2]}, rng, apply_errors=False)
    for i in range(1500):
        mujoco.mj_step(model, d)
        sens.record(model, d, None)
        if i % 2 == 1:
            e = est.update(sens.read(), 0.002)
    _, _, q, _ = truth(model, d)
    assert tilt_err_deg(e["torso_quat"], q) < 0.6     # from 3 deg


def test_output_keys_and_shapes(model):
    log, est, _ = run(model, CFG, stand_setup, 0.01, 0)
    e = log[-1]["e"]
    assert set(e) == set(ESTIMATE_KEYS)
    nj = est.n_joints
    assert e["com_vel"].shape == (3,) and e["torso_quat"].shape == (4,)
    assert e["torso_angvel"].shape == (3,) and e["foot_contact"].shape == (2,)
    assert e["joint_pos"].shape == (nj,) and e["joint_vel"].shape == (nj,)
    assert isinstance(e["free_fall"], bool)
    assert abs(np.linalg.norm(e["torso_quat"]) - 1.0) < 1e-9


def test_thruster_geometry_matches_truth(model):
    """With exact attitude and joints, (r, d) equal the true thruster geometry."""
    from suitsim.interfaces import load_thruster_specs
    log, est, d = run(model, CFG, fall_setup(1), 0.2, 0, noise_scale=0.0, apply_errors=False)
    specs = load_thruster_specs(CFG)
    c, _, q, _ = truth(model, d)
    jq = d.qpos[est.kin.hinge_qadr]
    e = {"torso_quat": q, "joint_pos": jq}
    r, dirs = est.thruster_geometry(e)
    sid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s.site) for s in specs]
    assert r.shape == (len(specs), 3) and dirs.shape == (len(specs), 3)
    np.testing.assert_allclose(r, d.site_xpos[sid] - c, atol=1e-9)
    np.testing.assert_allclose(dirs, d.site_xmat[sid].reshape(-1, 3, 3)[:, :, 2], atol=1e-9)
    # cached path (est dict from update) gives unit directions too
    r2, d2 = est.thruster_geometry(log[-1]["e"])
    np.testing.assert_allclose(np.linalg.norm(d2, axis=1), 1.0, atol=1e-9)


def test_estimator_uses_own_mjdata(model):
    _, est, d = run(model, CFG, stand_setup, 0.01, 0)
    assert est.kin.data is not d


def test_foot_contact_threshold_from_config(model):
    cfg = load_config(overrides={"estimation": {"foot_contact_threshold": 1e6}})
    log, _, _ = run(model, cfg, stand_setup, 0.05, 0)
    assert not log[-1]["e"]["foot_contact"].any()


def test_init_errors_applied(model):
    est = StateEstimator(model, CFG)
    q0 = np.array([1.0, 0, 0, 0])
    est.reset({"torso_quat": q0, "com_vel": np.zeros(3), "com_height": 1.0},
              np.random.default_rng(0))
    assert quat_err_deg(est.q, q0) == pytest.approx(CFG["estimation"]["init_attitude_err_deg"])
    assert np.linalg.norm(est.v) == pytest.approx(CFG["estimation"]["init_velocity_err"])
    assert abs(est.x[0] - 1.0) == pytest.approx(CFG["estimation"]["init_height_err"])


def test_update_timing(model):
    d = mujoco.MjData(model)
    stand_setup(model, d)
    sens = SensorSuite(model, CFG, np.random.default_rng(0))
    sens.reset(model, d)
    est = StateEstimator(model, CFG)
    s = sens.read()
    for _ in range(50):
        est.update(s, 0.002)
    n = 1000
    t0 = time.perf_counter()
    for _ in range(n):
        est.update(s, 0.002)
    per = (time.perf_counter() - t0) / n
    assert per < 0.5e-3, per     # target ~0.1 ms; lenient for slow CI machines
