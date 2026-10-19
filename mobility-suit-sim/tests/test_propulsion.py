"""Hardware-realistic propulsion options: multi-base configs, fan-less suits, propellant mass
depletion, on/off valves (sigma-delta) and solid pulse charges."""
import math

import mujoco
import numpy as np
import pytest
import yaml

from suitsim.actuators import ThrusterBank
from suitsim.actuators.propellant import PropellantTracker
from suitsim.actuators.thrusters import fan_k
from suitsim.envs import make_env
from suitsim.eval.metrics import budgets_from_cfg, summarize
from suitsim.interfaces import (
    BUDGET_KEYS, G0, PropellantSpec, budget_capacity, deep_merge, load_config,
    load_thruster_specs, propellant_specs, suit_mass_items,
)
from suitsim.model import build_model, model_info

DT = 0.001


@pytest.fixture(scope="module")
def rcfg():
    return load_config("suit_realistic.yaml")


def make_bank(cfg):
    return ThrusterBank(load_thruster_specs(cfg), cfg, cfg["sim"]["timestep"])


def run(bank, u, steps):
    return np.array([bank.step(u) for _ in range(steps)])


# ------------------------------------------------------------------------- config loading
def test_load_config_list_base(tmp_path):
    (tmp_path / "a.yaml").write_text(yaml.safe_dump({"x": 1, "d": {"p": 1, "q": 1}}))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump({"base": "a.yaml", "d": {"q": 2}, "y": 2}))
    (tmp_path / "c.yaml").write_text(yaml.safe_dump({"d": {"p": 3}, "z": 3}))
    (tmp_path / "child.yaml").write_text(yaml.safe_dump({"base": ["b.yaml", "c.yaml"], "z": 4}))
    # single-string base still works (b -> a)
    assert load_config(tmp_path / "b.yaml") == {"x": 1, "d": {"p": 1, "q": 2}, "y": 2}
    # list: merged in order, child on top
    assert load_config(tmp_path / "child.yaml") == {"x": 1, "y": 2, "z": 4,
                                                   "d": {"p": 3, "q": 2}}


@pytest.mark.parametrize("task", ["drop", "jump", "push"])
@pytest.mark.parametrize("suit", ["suit_realistic", "suit_realistic_peroxide"])
def test_realistic_task_configs_compose(task, suit):
    """task_<t>_realistic*.yaml == the task with the realistic suit: nothing of either lost."""
    variant = "realistic" if suit == "suit_realistic" else "realistic_peroxide"
    cfg = load_config(f"task_{task}_{variant}.yaml")
    t = load_config(f"task_{task}.yaml")
    s = load_config(f"{suit}.yaml")
    assert cfg == deep_merge(t, s)
    assert cfg["suit"] == s["suit"]
    for k in t:                       # lossless: every shared non-suit section is identical
        if k != "suit":
            assert cfg[k] == t[k], k


# ------------------------------------------------------------------------- no fans
def test_realistic_suit_has_no_fans(rcfg):
    specs = load_thruster_specs(rcfg)
    assert len(specs) == 42 and not any(s.cls == "fan" for s in specs)
    assert fan_k(rcfg) == 0.0
    bank = make_bank(rcfg)
    assert bank.idx_fans.size == 0 and bank.fan_power_w().size == 0
    run(bank, bank.f_max.copy(), 50)
    assert bank.used()["battery_wh"] == 0.0
    assert bank.telemetry()["class_force"]["fan"] == 0.0


def test_null_thruster_and_count_zero(rcfg):
    c = load_config("suit_default.yaml", {"suit": {"thrusters": {"fan_l": None}}})
    assert len(load_thruster_specs(c)) == 43
    c = load_config("suit_default.yaml", {"suit": {"actuator_classes": {"fan": {"count": 0}}}})
    assert len(load_thruster_specs(c)) == 42


def test_realistic_masses(rcfg):
    m = build_model(rcfg)
    info = model_info(m, rcfg)
    ps = {p.budget: p for p in propellant_specs(rcfg)}
    assert ps["impulse_gas_jets"].mass == pytest.approx(600.0 / (70.0 * G0))       # 0.874 kg
    assert ps["impulse_gas_pods"].mass == pytest.approx(2496.0 / (200.0 * G0))     # 1.273 kg
    dry = sum(v["total"] for v in rcfg["suit"]["masses"].values() if v)
    assert dry == pytest.approx(13.0)
    assert info["suit_mass"] == pytest.approx(dry + 0.874 + 1.273, abs=2e-3)
    assert info["total_mass"] == pytest.approx(91.0 + info["suit_mass"], abs=1e-3)
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    want = 91.0 * rcfg["human"]["mass_frac"]["torso"] + sum(
        mm for _, b, mm in suit_mass_items(rcfg) if b == "torso")
    assert m.body_mass[torso] == pytest.approx(want, rel=1e-6)
    per = load_config("suit_realistic_peroxide.yaml")
    pp = {p.budget: p for p in propellant_specs(per)}["impulse_gas_pods"]
    assert pp.isp == 130.0 and pp.mass == pytest.approx(1.961, abs=1e-3)
    assert "solid_pulses" not in (per["suit"]["actuator_classes"]["lift_pod"] or {}) or \
        per["suit"]["actuator_classes"]["lift_pod"]["solid_pulses"] is None
    assert make_bank(per).idx_pulse.size == 0


def test_default_suit_unchanged():
    cfg = load_config()
    assert propellant_specs(cfg) == []
    assert all(budget_capacity(cfg, k) == float(cfg["suit"]["budgets"][k]) for k in BUDGET_KEYS)
    bank = make_bank(cfg)
    assert bank.idx_valve.size == 0 and bank.idx_pulse.size == 0
    assert budgets_from_cfg(cfg) == {k: float(cfg["suit"]["budgets"][k]) for k in BUDGET_KEYS}


# ------------------------------------------------------------------------- propellant mass
_ROCKET_XML = """<mujoco><option gravity="0 0 0" timestep="{dt}"/><worldbody>
<body name="a"><freejoint/><geom type="sphere" size="0.1" mass="{dry}"/>
  <geom name="prop" type="sphere" size="0.02" mass="{prop}" contype="0" conaffinity="0"/>
  <body name="b" pos="0.5 0 0"><joint type="hinge" axis="0 0 1"/>
    <geom type="sphere" size="0.05" mass="1" contype="0" conaffinity="0"/>
    <geom name="prop_b" type="sphere" size="0.02" mass="{prop}" contype="0" conaffinity="0"/></body>
</body></worldbody></mujoco>"""


def test_rocket_equation():
    """A free body losing propellant under constant thrust follows the rocket equation
    (MuJoCo picks up runtime body_mass changes in the dynamics, no mj_setConst)."""
    dry, prop, F, isp = 10.0, 5.0, 100.0, 10.0
    m = mujoco.MjModel.from_xml_string(_ROCKET_XML.format(dt=DT, dry=dry, prop=prop))
    d = mujoco.MjData(m)
    # only body "a" burns (budget pods); the child keeps its propellant (budget jets unused)
    tr = PropellantTracker(m, [PropellantSpec("impulse_gas_pods", "a", isp, prop),
                               PropellantSpec("impulse_gas_jets", "b", isp, prop)], 0.02)
    m0 = float(m.body_subtreemass[1])
    c = isp * G0
    a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "a")
    d.qpos[:] = m.qpos0
    T = 4.0
    n = int(round(T / DT))
    used = 0.0
    for _ in range(n):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        d.xfrc_applied[:] = 0.0
        d.xfrc_applied[a, :3] = [F, 0.0, 0.0]
        # force at the body CoM; the hinge child is pulled along rigidly (no torque since
        # the subtree is symmetric along x and the joint is about z: CoM on the x axis)
        used += F * DT
        tr.update({"impulse_gas_pods": used})
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    mujoco.mj_subtreeVel(m, d)
    m1 = m0 - used / c
    v_rocket = c * math.log(m0 / m1)
    v = float(d.subtree_linvel[a][0])
    assert tr.used()["impulse_gas_pods"] == pytest.approx(used / c)
    assert m.body_subtreemass[a] == pytest.approx(m1, rel=1e-12)
    assert v == pytest.approx(v_rocket, rel=0.01)
    assert abs(v - F * T / m0) / v_rocket > 0.10            # constant-mass answer is far off
    # subtree constants consistent with a full mj_setConst recomputation
    sub = m.body_subtreemass.copy()
    m2 = mujoco.MjModel.from_xml_string(_ROCKET_XML.format(dt=DT, dry=dry, prop=prop))
    m2.body_mass[:] = m.body_mass
    m2.body_inertia[:] = m.body_inertia
    mujoco.mj_setConst(m2, mujoco.MjData(m2))
    np.testing.assert_allclose(sub, m2.body_subtreemass, rtol=1e-12)
    # inertia share of the propellant sphere removed in proportion
    assert m.body_inertia[a, 0] == pytest.approx(tr._inertia[a, 0] - 0.4 * 0.02 ** 2 * used / c)
    tr.restore()
    assert m.body_subtreemass[a] == pytest.approx(m0)


def test_env_propellant_depletion_and_reset():
    cfg = load_config("task_drop_realistic.yaml", {"task": {"fixed": {"drop_height": 3.0}}})
    env = make_env("drop_landing", cfg, controller="baseline")
    pel = env.info_m["body_ids"]["pelvis"]
    tor = env.info_m["body_ids"]["torso"]
    infos, masses = [], []
    for _ in range(2):
        env.reset(seed=7)
        m_start = env.model.body_mass.copy()
        sub0 = float(env.model.body_subtreemass[env.root_id])
        assert sub0 == pytest.approx(env.params["total_mass"])
        # composes with the randomiser: human mass scaled, suit + full propellant kept
        assert sub0 == pytest.approx(91.0 * env.params["mass_scale"] + env.info_m["suit_mass"],
                                     abs=1e-4)          # MJCF masses carry 6 significant digits
        done = False
        while not done:
            _, _, te, tr, info = env.step(None)
            done = te or tr
        used = env.thrusters.used()
        dm_p = used["impulse_gas_pods"] / (200.0 * G0)
        dm_j = used["impulse_gas_jets"] / (70.0 * G0)
        assert dm_p > 0.1 and dm_j > 0.05
        assert env.model.body_mass[pel] == pytest.approx(m_start[pel] - dm_p, abs=1e-9)
        assert env.model.body_mass[tor] == pytest.approx(m_start[tor] - dm_j, abs=1e-9)
        assert info["propellant_kg"] == pytest.approx(dm_p + dm_j)
        assert info["total_mass_end"] == pytest.approx(sub0 - dm_p - dm_j, abs=1e-9)
        infos.append(info)
        masses.append(m_start)
    # reset refills: identical start masses and bit-identical episode
    np.testing.assert_array_equal(masses[0], masses[1])
    for k in ("impact_v_vertical", "impulse_gas_pods", "impulse_gas_jets", "propellant_kg",
              "pulse_charges_used", "valve_pulses"):
        assert infos[0][k] == infos[1][k], k
    s = summarize(infos, cfg)
    assert s["stats"]["propellant_kg"]["n"] == 2
    assert s["budgets"]["impulse_gas_pods"] == 2496.0


# ------------------------------------------------------------------------- on/off valves
def _valve_cfg(rcfg, **valve):
    return deep_merge(rcfg, {"suit": {"actuator_classes": {"micro_jet": {"valve": valve}}}})


@pytest.mark.parametrize("u0", [3.1, 8.0, 13.7, 20.0, 27.3, 36.0, 39.2])
def test_valve_tracks_constant_command(rcfg, u0):
    """Sliding 100 ms windows: within 5 % for u >= f_max/2.  Below that the minimum pulse
    (P = f_max * min_on_time = 0.2 N*s) makes a window ripple of up to P (1 - u/f_max) / T
    unavoidable (any on/off modulator), so the test checks that bound, plus the 1 s mean."""
    bank = make_bank(rcfg)
    j = bank.idx_jets[0]
    fm = bank.f_max[j]
    u = np.zeros(bank.n)
    u[j] = u0
    F = run(bank, u, 1100)[:, j]
    assert np.all(F <= fm + 1e-12) and np.all(F >= 0)
    T = 0.1
    P = fm * rcfg["suit"]["actuator_classes"]["micro_jet"]["valve"]["min_on_time"]
    tol = 0.05 * u0 if u0 >= 0.5 * fm else P * (1 - u0 / fm) / T + fm * DT / T
    for s in range(30, 1000, 3):                      # every 100 ms window after warm-up
        assert abs(F[s:s + 100].mean() - u0) <= tol
    assert F[100:1100].mean() == pytest.approx(u0, rel=0.01, abs=P / 1.0)
    # impulse accounting: the jets' budget is exactly the delivered impulse
    assert bank.used()["impulse_gas_jets"] == pytest.approx(F.sum() * DT, rel=1e-12)


def test_valve_square_pulses_delays_and_min_on(rcfg):
    cfg = _valve_cfg(rcfg, rise_tau=0.0)
    bank = make_bank(cfg)
    j = bank.idx_jets[0]
    fm = bank.f_max[j]
    u = np.zeros(bank.n)
    u0 = 4.4                                          # ~11 % duty -> minimum-length pulses
    u[j] = u0
    F = run(bank, u, 400)[:, j]
    assert set(np.unique(F)) <= {0.0, fm}             # strictly on/off
    on = F > 0
    first = int(np.argmax(on))
    # the modulator fires once e = (k+1) u dt > 0.5 f dt; the valve opens open_delay later
    n_open = int(round(cfg["suit"]["actuator_classes"]["micro_jet"]["valve"]["open_delay"] / DT))
    fire = int(np.ceil(0.5 * fm * DT / (u0 * DT))) - 1
    assert first == fire + n_open
    runs, cur = [], 0
    for x in on[first:]:
        if x:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    assert len(runs) >= 3 and min(runs) >= 5          # min_on_time 5 ms
    assert np.all(F[: first] == 0)
    # the valves feed only the jets' budget
    assert bank.used()["impulse_gas_pods"] == 0.0
    assert bank.valve_pulses() == pytest.approx(F.sum() * DT / (fm * 0.005), abs=1)


def test_valve_asymmetric_delay_compensated(rcfg):
    cfg = _valve_cfg(rcfg, open_delay=0.002, close_delay=0.005, rise_tau=0.0)
    bank = make_bank(cfg)
    j = bank.idx_jets[0]
    u = np.zeros(bank.n)
    u[j] = 10.0
    F = run(bank, u, 1500)[:, j]
    assert F[100:1100].mean() == pytest.approx(10.0, rel=0.02)


def test_valve_zero_command_never_fires(rcfg):
    bank = make_bank(rcfg)
    F = run(bank, np.zeros(bank.n), 200)
    assert np.all(F == 0) and bank.used()["impulse_gas_jets"] == 0.0


# ------------------------------------------------------------------------- solid pulses
def test_solid_pulse_auto_scheduler(rcfg):
    bank = make_bank(rcfg)
    sp = rcfg["suit"]["actuator_classes"]["lift_pod"]["solid_pulses"]
    I, tb, td = sp["charge_impulse"], sp["burn_time"], sp["ignition_delay"]
    Fb = I / tb
    p0, p1 = bank.idx_pulse
    assert list(bank.idx_pulse) == list(bank.idx_pods)
    assert bank.capacity["impulse_gas_pods"] == pytest.approx(2 * 8 * I)
    u = np.zeros(bank.n)
    u[p0] = 1500.0                                     # full demand on pod 0 only
    F = run(bank, u, 300)
    # first ignition when the accumulated command reaches trigger_fraction * I
    n_trig = int(np.ceil(sp["trigger_fraction"] * I / (1500.0 * DT)))
    t_ign = bank.ignition_log()[0][0]
    assert t_ign == pytest.approx((n_trig - 1) * DT)
    first = int(np.argmax(F[:, p0] > 0))
    assert first == int(round((t_ign + td) / DT))
    assert np.all(F[:, p1] == 0)
    assert set(np.round(np.unique(F[:, p0]), 9)) <= {0.0, round(Fb, 9)}
    # pre-ignition: back-to-back charges burn without a gap
    burning = F[first:, p0] > 0
    assert burning.all()
    # cannot be stopped: zero command, the burning charge finishes, nothing new ignites
    n_before = bank.charges_used()
    F2 = run(bank, np.zeros(bank.n), 400)
    assert bank.charges_used() == n_before
    total = bank.used()["impulse_gas_pods"]
    assert total == pytest.approx(n_before * I, rel=1e-9)            # whole charges only
    assert F2[-1, p0] == 0.0
    assert (F[:, p0].sum() + F2[:, p0].sum()) * DT == pytest.approx(total, rel=1e-12)


def test_solid_pulse_tracks_average_and_budget_exhaustion(rcfg):
    bank = make_bank(rcfg)
    u = np.zeros(bank.n)
    u[bank.idx_pulse] = 600.0
    F = run(bank, u, 2000)[:, bank.idx_pulse]
    # 2 s at 600 N per pod = 1200 N*s each, i.e. ~7.7 charges -> 8 per pod at most
    per_pod = F.sum(axis=0) * DT
    assert np.all(np.abs(per_pod - 1200.0) <= 156.0 + 1e-6)
    run(bank, u, 3000)
    assert np.all(bank.charges_remaining() == 0)
    assert np.all(bank.u_max[bank.idx_pulse] == 0)
    assert bank.used()["impulse_gas_pods"] == pytest.approx(2496.0, rel=1e-9)
    assert bank.remaining()["impulse_gas_pods"] == pytest.approx(0.0, abs=1e-6)
    assert np.all(run(bank, u, 10)[:, bank.idx_pulse] == 0)


def test_solid_pulse_manual_api(rcfg):
    cfg = deep_merge(rcfg, {"suit": {"actuator_classes": {"lift_pod": {
        "solid_pulses": {"scheduler": "manual"}}}}})
    bank = make_bank(cfg)
    sp = cfg["suit"]["actuator_classes"]["lift_pod"]["solid_pulses"]
    u = np.zeros(bank.n)
    u[bank.idx_pulse] = 1500.0                          # ignored in manual mode
    assert np.all(run(bank, u, 50)[:, bank.idx_pulse] == 0)
    assert bank.ignite(1) is True
    assert bank.ignite(1) is False                       # one charge at a time (max_concurrent)
    F = run(bank, np.zeros(bank.n), 200)[:, bank.idx_pulse]
    assert np.all(F[:, 0] == 0)
    assert F[:, 1].sum() * DT == pytest.approx(sp["charge_impulse"], rel=1e-12)
    assert int(np.argmax(F[:, 1] > 0)) == int(round(sp["ignition_delay"] / DT))
    assert bank.charges_remaining().tolist() == [8, 7]
    for _ in range(7):
        assert bank.ignite(1)
        run(bank, np.zeros(bank.n), 120)
    assert bank.ignite(1) is False                       # out of charges
    with pytest.raises(IndexError):
        bank.ignite(2)


def test_solid_pulse_dr_scales_keep_charge_impulse(rcfg):
    bank = make_bank(rcfg)
    sf = np.full(bank.n, 1.15)
    st = np.full(bank.n, 1.5)
    bank.reset(scale_fmax=sf, scale_tau=st)
    bank.ignite(0)
    F = run(bank, np.zeros(bank.n), 200)[:, bank.idx_pulse[0]]
    assert F.max() == pytest.approx(1500.0 * 1.15)
    assert F.sum() * DT == pytest.approx(156.0, rel=1e-12)
    assert int(np.argmax(F > 0)) == int(round(0.010 * 1.5 / DT))
