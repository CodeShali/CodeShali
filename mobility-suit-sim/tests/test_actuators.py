import mujoco
import numpy as np
import pytest

from suitsim.actuators import ThrusterBank, apply_thrusters, body_drag_wrench, fan_power
from suitsim.interfaces import BUDGET_KEYS, ThrusterBankProtocol, load_config, load_thruster_specs


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture
def bank(cfg):
    return ThrusterBank(load_thruster_specs(cfg), cfg, cfg["sim"]["timestep"])


def full(bank, idx, frac=1.0):
    u = np.zeros(bank.n)
    u[idx] = bank.f_max[idx] * frac
    return u


def run(bank, u, seconds):
    for _ in range(int(round(seconds / bank.dt))):
        f = bank.step(u)
    return f


def test_layout(bank, cfg):
    assert isinstance(bank, ThrusterBankProtocol)
    assert bank.n == 44
    assert len(bank.idx_pods) == 2 and len(bank.idx_fans) == 2 and len(bank.idx_jets) == 40
    np.testing.assert_array_equal(bank.class_index("fan"), bank.idx_fans)
    cl = cfg["suit"]["actuator_classes"]
    assert np.all(bank.f_max[bank.idx_pods] == cl["lift_pod"]["f_max"])
    assert np.all(bank.f_max[bank.idx_jets] == cl["micro_jet"]["f_max"])
    assert set(bank.remaining()) == set(BUDGET_KEYS)
    assert bank.remaining() == {k: float(cfg["suit"]["budgets"][k]) for k in BUDGET_KEYS}


def test_lag_step_response(bank):
    i = bank.idx_jets[0]
    u = full(bank, [i])
    tau, dt, fm = bank.tau[i], bank.dt, bank.f_max[i]
    f = run(bank, u, tau)
    n = int(round(tau / dt))
    expected = 1 - (1 - dt / tau) ** n          # forward-Euler lag
    assert f[i] / fm == pytest.approx(expected, rel=1e-9)
    assert 0.60 < f[i] / fm < 0.70             # ~63 % after one tau
    f = run(bank, u, 10 * tau)
    assert f[i] == pytest.approx(fm, rel=1e-3)


def test_fan_slower_than_jet(bank):
    u = full(bank, [bank.idx_fans[0], bank.idx_jets[0]])
    f = run(bank, u, 0.01)
    assert f[bank.idx_jets[0]] / bank.f_max[bank.idx_jets[0]] > 0.8
    assert f[bank.idx_fans[0]] / bank.f_max[bank.idx_fans[0]] < 0.15
    f = run(bank, u, 0.09)                      # total 0.1 s = one fan tau
    assert 0.60 < f[bank.idx_fans[0]] / bank.f_max[bank.idx_fans[0]] < 0.70


def test_clamping(bank):
    u = np.full(bank.n, -100.0)
    f = run(bank, u, 0.05)
    assert np.all(f == 0.0) and np.all(bank.commands == 0.0)
    u = np.full(bank.n, 1e6)
    f = run(bank, u, 0.05)
    assert np.all(f <= bank.f_max + 1e-12) and np.all(f >= 0)
    np.testing.assert_allclose(bank.commands, bank.f_max)
    assert f[bank.idx_jets[0]] == pytest.approx(bank.f_max[bank.idx_jets[0]], rel=1e-3)


def test_bad_shape(bank):
    with pytest.raises(ValueError):
        bank.step(np.zeros(3))


def test_budget_deduction_exact(bank):
    # alpha = 1 (tau tiny) so the force is exactly the command from the first step
    bank.reset(scale_tau=np.full(bank.n, 1e-6))
    u = np.zeros(bank.n)
    u[bank.idx_pods] = 500.0
    u[bank.idx_jets[:4]] = 10.0
    run(bank, u, 1.0)
    used = bank.used()
    assert used["impulse_gas_pods"] == pytest.approx(2 * 500.0 * 1.0, rel=1e-9)
    assert used["impulse_gas_jets"] == pytest.approx(4 * 10.0 * 1.0, rel=1e-9)
    assert used["battery_wh"] == 0.0
    tot = bank.remaining()["impulse_gas_pods"] + used["impulse_gas_pods"]
    assert tot == pytest.approx(bank.capacity["impulse_gas_pods"])


def test_pod_budget_shared_and_exhausted(bank, cfg):
    cap = cfg["suit"]["budgets"]["impulse_gas_pods"]
    bank.reset(scale_tau=np.full(bank.n, 1e-6))
    u = np.zeros(bank.n)
    u[bank.idx_pods[0]] = 1500.0                 # only pod_l fires
    t_empty = cap / 1500.0                       # 1.667 s
    run(bank, u, t_empty + 0.1)
    assert bank.remaining()["impulse_gas_pods"] == 0.0
    assert bank.used()["impulse_gas_pods"] == pytest.approx(cap)
    # pod_r never fired, but it shares the tank -> also dead
    u[bank.idx_pods] = 1500.0
    f = bank.step(u)
    assert np.all(f[bank.idx_pods] == 0.0)
    assert np.all(bank.u_max[bank.idx_pods] == 0.0)
    assert np.all(bank.u_max[bank.idx_jets] > 0.0)
    assert bank.remaining()["impulse_gas_pods"] >= 0.0


def test_partial_last_step(bank):
    bank.reset(scale_tau=np.full(bank.n, 1e-6))
    bank._rem["impulse_gas_pods"] = 1.0          # 1 N*s left
    u = full(bank, bank.idx_pods)                # 2*1500*1ms = 3 N*s demanded
    f = bank.step(u)
    assert np.sum(f[bank.idx_pods]) * bank.dt == pytest.approx(1.0)
    assert bank.remaining()["impulse_gas_pods"] == 0.0
    assert np.all(bank.step(u)[bank.idx_pods] == 0.0)


def test_jet_budget_shared_across_40(bank, cfg):
    cap = cfg["suit"]["budgets"]["impulse_gas_jets"]
    bank.reset(scale_tau=np.full(bank.n, 1e-6))
    u = full(bank, bank.idx_jets)                # 40*40 = 1600 N -> empty in 0.375 s
    run(bank, u, cap / 1600.0 + 0.05)
    assert bank.remaining()["impulse_gas_jets"] == 0.0
    assert bank.used()["impulse_gas_jets"] == pytest.approx(cap)
    assert np.all(bank.u_max[bank.idx_jets] == 0.0)
    assert np.all(bank.forces[bank.idx_jets] == 0.0)
    assert np.all(bank.u_max[bank.idx_pods] > 0.0)


def test_fan_power(bank, cfg):
    assert fan_power(300.0, cfg) == pytest.approx(cfg["suit"]["budgets"]["fan_peak_power_w"])
    assert fan_power(300.0, bank.k_fan) == pytest.approx(1200.0)
    assert fan_power(150.0, cfg) == pytest.approx(1200.0 / 2 ** 1.5)
    assert fan_power(-5.0, cfg) == 0.0


def test_battery_drain_10s(bank):
    bank.reset(scale_tau=np.full(bank.n, 1e-6))
    run(bank, full(bank, bank.idx_fans), 10.0)
    expected = 2 * 1200.0 * 10.0 / 3600.0        # 6.667 Wh
    assert bank.used()["battery_wh"] == pytest.approx(expected, rel=1e-9)
    np.testing.assert_allclose(bank.telemetry()["fan_power_w"], 1200.0)


def test_battery_exhaustion(bank):
    bank.reset(scale_tau=np.full(bank.n, 1e-6))
    bank._rem["battery_wh"] = 0.5               # 2 fans at 2.4 kW -> 0.75 s
    run(bank, full(bank, bank.idx_fans), 1.0)
    assert bank.remaining()["battery_wh"] == 0.0
    assert np.all(bank.forces[bank.idx_fans] == 0.0)
    assert np.all(bank.u_max[bank.idx_fans] == 0.0)


def test_reset_with_scales(bank):
    run(bank, full(bank, np.arange(bank.n)), 0.5)
    sf = np.linspace(0.85, 1.15, bank.n)
    st = np.linspace(0.5, 1.5, bank.n)
    bank.reset(scale_fmax=sf, scale_tau=st)
    assert np.all(bank.forces == 0.0)
    assert bank.remaining() == bank.capacity
    np.testing.assert_allclose(bank.f_max, bank.f_max_nom * sf)
    np.testing.assert_allclose(bank.u_max, bank.f_max_nom * sf)
    np.testing.assert_allclose(bank.tau, bank.tau_nom * st)
    f = run(bank, np.full(bank.n, 1e6), 0.2)   # < jet-budget exhaustion
    np.testing.assert_allclose(f[bank.idx_jets], bank.f_max[bank.idx_jets], rtol=1e-6)
    k_nom = bank.k_fan
    bank.reset(scale_fmax=np.full(bank.n, 1.15))
    assert bank.k_fan == k_nom
    bank.reset()
    np.testing.assert_allclose(bank.f_max, bank.f_max_nom)


def test_telemetry(bank):
    u = full(bank, np.arange(bank.n))
    run(bank, u, 0.05)
    t = bank.telemetry()
    assert set(t) >= {"forces", "commands", "remaining", "used", "class_force", "fan_power_w"}
    assert t["class_force"]["micro_jet"] == pytest.approx(bank.forces[bank.idx_jets].sum())
    t["forces"][:] = -1
    assert np.all(bank.forces >= 0)


# --------------------------------------------------------------------------- MuJoCo
MJCF = """
<mujoco>
  <option timestep="0.001" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="b" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" mass="{mass}" pos="0.02 0 0"/>
      <site name="up" pos="0.12 0.0 0.0" zaxis="0 0 1"/>
      <site name="side" pos="0.0 0.05 0.03" zaxis="1 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_apply_thrusters_torque_and_physics():
    mass = 2.0
    model = mujoco.MjModel.from_xml_string(MJCF.format(mass=mass))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in ("up", "side")]
    bid = model.site_bodyid[sid]
    F_up = 3.0 * mass * 9.81
    w = apply_thrusters(model, data, sid, bid, np.array([F_up, 0.0]))
    assert w.shape == (model.nbody, 6)
    b = bid[0]
    com = data.xipos[b]
    np.testing.assert_allclose(com, [0.02, 0, 1], atol=1e-12)
    np.testing.assert_allclose(w[b, :3], [0, 0, F_up])
    r = np.array([0.12 - 0.02, 0, 0])
    np.testing.assert_allclose(w[b, 3:], np.cross(r, [0, 0, F_up]))   # (0, -0.1 F, 0)
    assert w[b, 4] == pytest.approx(-0.1 * F_up)

    # two thrusters on one body accumulate; cross-check with mj_applyFT
    f = np.array([F_up, 7.0])
    w = apply_thrusters(model, data, sid, None, f)
    q_ref = np.zeros(model.nv)
    for s, fi in zip(sid, f):
        Fw = fi * data.site_xmat[s].reshape(3, 3)[:, 2]
        mujoco.mj_applyFT(model, data, Fw, np.zeros(3), data.site_xpos[s], b, q_ref)
    q_got = np.zeros(model.nv)
    mujoco.mj_applyFT(model, data, w[b, :3].copy(), w[b, 3:].copy(), com.copy(), b, q_got)
    np.testing.assert_allclose(q_got, q_ref, atol=1e-10)
    np.testing.assert_allclose(w[b, :3], [7.0, 0, F_up])

    # out reuse: zeroed unless accumulate
    out = np.ones((model.nbody, 6))
    apply_thrusters(model, data, sid, bid, np.zeros(2), out=out)
    assert np.all(out == 0)
    apply_thrusters(model, data, sid, bid, f, out=out)
    apply_thrusters(model, data, sid, bid, f, out=out, accumulate=True)
    np.testing.assert_allclose(out, 2 * w)

    # physics: thrust through the CoM line (pure lift) > mg -> accelerates up at ~2g
    data.xfrc_applied[:] = 0
    data.xfrc_applied[b, :3] = [0, 0, F_up]
    for _ in range(100):
        mujoco.mj_step(model, data)
    vz = data.qvel[2]
    assert vz == pytest.approx((F_up / mass - 9.81) * 0.1, rel=1e-3)
    assert data.qpos[2] > 1.0

    # offset thruster -> pitch (about -y for +x offset) and upward acceleration
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    for _ in range(50):
        w = apply_thrusters(model, data, sid[:1], bid[:1], np.array([F_up]))
        data.xfrc_applied[:] = w
        mujoco.mj_step(model, data)
    assert data.qvel[2] > 0
    assert data.qvel[4] < 0                      # world angular velocity about y


def test_body_drag(cfg):
    model = mujoco.MjModel.from_xml_string(MJCF.format(mass=2.0))
    data = mujoco.MjData(model)
    data.qvel[:3] = [3.0, 0.0, -4.0]             # |v| = 5
    mujoco.mj_forward(model, data)
    F = body_drag_wrench(model, data, cfg, 1)
    s = cfg["sim"]
    c = 0.5 * s["air_density"] * s["drag_cd"] * s["drag_area"]
    np.testing.assert_allclose(F, -c * 5.0 * np.array([3.0, 0.0, -4.0]), rtol=1e-9)
    assert F[2] > 0                              # opposes the fall
    np.testing.assert_allclose(body_drag_wrench(model, data, cfg, 1, com_vel=np.zeros(3)), 0)
