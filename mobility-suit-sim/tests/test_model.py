import mujoco
import numpy as np
import pytest

from suitsim.interfaces import (
    AIRBAGS, ALTIMETER_SITE, FEET, GROUND_GEOM, SEGMENTS, airbag_geom, foot_sole_site,
    foot_touch_sensor, imu_accel_sensor, imu_gyro_sensor, imu_site, load_config,
    load_thruster_specs, thruster_site,
)
from suitsim.model import (
    AirbagState, ModelRandomizer, airbag_fraction, airbags_deployed, apply_mass_scale, build_mjcf, build_model, com_state,
    lowest_point, model_info, set_airbags, set_ground_friction, set_initial_pose,
)

UP = np.array([1.0, 0.0, 0.0, 0.0])


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture()
def model(cfg):
    return build_model(cfg)


def _id(m, obj, name):
    return mujoco.mj_name2id(m, obj, name)


def test_mass(model, cfg):
    info = model_info(model, cfg)
    assert abs(info["total_mass"] - 100.0) < 0.5
    assert abs(info["human_mass"] - cfg["human"]["mass"]) < 1e-3
    for seg in SEGMENTS:
        b = _id(model, mujoco.mjtObj.mjOBJ_BODY, seg)
        suit = sum(v["total"] / len(v["bodies"]) for v in cfg["suit"]["masses"].values()
                   if seg in v["bodies"])
        want = cfg["human"]["mass"] * cfg["human"]["mass_frac"][seg] + suit
        assert model.body_mass[b] == pytest.approx(want, rel=1e-4)


def test_segments_and_tree(model, cfg):
    for seg in SEGMENTS:
        b = _id(model, mujoco.mjtObj.mjOBJ_BODY, seg)
        assert b > 0
        parent = cfg["human"]["skeleton"][seg]["parent"]
        pname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.body_parentid[b])
        assert pname == parent
    assert model.nbody == len(SEGMENTS) + 1


def test_joints_and_servos(model, cfg):
    info = model_info(model, cfg)
    assert info["joint_names"] == list(cfg["human"]["joints"]["dofs"])
    assert np.all(np.diff(info["qpos_adr"]) > 0)
    assert np.all(info["servo_ids"] >= 0)
    assert model.nu == len(info["joint_names"])


def test_thruster_sites(model, cfg):
    d = mujoco.MjData(model)
    mujoco.mj_kinematics(model, d)   # qpos0: identity pose for every body
    specs = load_thruster_specs(cfg)
    assert len(specs) == 44
    info = model_info(model, cfg)
    for s, sid in zip(specs, info["thruster_site_ids"]):
        assert sid == _id(model, mujoco.mjtObj.mjOBJ_SITE, thruster_site(s.name))
        z = d.site_xmat[sid].reshape(3, 3)[:, 2]
        np.testing.assert_allclose(z, s.dir, atol=1e-6)
        b = _id(model, mujoco.mjtObj.mjOBJ_BODY, s.body)
        np.testing.assert_allclose(d.site_xpos[sid], d.xpos[b] + s.pos, atol=1e-9)


def test_sensors(model):
    for seg in SEGMENTS:
        assert _id(model, mujoco.mjtObj.mjOBJ_SITE, imu_site(seg)) >= 0
        assert _id(model, mujoco.mjtObj.mjOBJ_SENSOR, imu_accel_sensor(seg)) >= 0
        assert _id(model, mujoco.mjtObj.mjOBJ_SENSOR, imu_gyro_sensor(seg)) >= 0
    for f in FEET:
        assert _id(model, mujoco.mjtObj.mjOBJ_SITE, foot_sole_site(f)) >= 0
        assert _id(model, mujoco.mjtObj.mjOBJ_SENSOR, foot_touch_sensor(f)) >= 0
    assert _id(model, mujoco.mjtObj.mjOBJ_SITE, ALTIMETER_SITE) >= 0
    assert _id(model, mujoco.mjtObj.mjOBJ_GEOM, GROUND_GEOM) >= 0


def test_airbags_toggle(model, cfg):
    ids = [_id(model, mujoco.mjtObj.mjOBJ_GEOM, airbag_geom(b)) for b in AIRBAGS]
    assert all(i >= 0 for i in ids)
    c = cfg["human"]["contact"]
    ac = cfg["suit"]["airbags"]
    # compiled colliding with the ground (MuJoCo fixes collision filtering at compile time)
    for g in ids:
        assert model.geom_contype[g] == c["body_contype"]
        assert model.geom_conaffinity[g] == c["body_conaffinity"]
    # compiled stowed: tiny, inside the carrier segment, invisible
    assert not airbags_deployed(model)
    assert airbag_fraction(model) == pytest.approx(0.0)
    for bag, g in zip(AIRBAGS, ids):
        assert model.geom_rgba[g, 3] == 0
        dep = np.asarray(ac[bag]["size"], float)
        np.testing.assert_allclose(model.geom_size[g, :dep.size], ac["stowed_scale"] * dep)
        assert model.geom_rbound[g] < 0.01
    set_airbags(model, True)
    for bag, g in zip(AIRBAGS, ids):
        dep = np.asarray(ac[bag]["size"], float)
        np.testing.assert_allclose(model.geom_size[g, :dep.size], dep)
        np.testing.assert_allclose(model.geom_pos[g], ac[bag]["pos"])
        assert model.geom_rgba[g, 3] > 0
        assert model.geom_contype[g] == c["body_contype"]
    assert airbags_deployed(model)
    mujoco.mj_setConst(model, mujoco.MjData(model))       # used to fail: 'setBound: ...'
    assert airbags_deployed(model)
    set_airbags(model, 0.5)
    assert airbag_fraction(model) == pytest.approx(0.5)
    set_airbags(model, False)
    assert not airbags_deployed(model)
    mujoco.mj_setConst(model, mujoco.MjData(model))


def test_airbag_inflation_over_deploy_time(model, cfg):
    st = AirbagState(model, cfg)
    st.stow(model)
    dt = float(model.opt.timestep)
    st.step(model, dt)
    assert airbag_fraction(model, cfg) == 0.0              # not triggered
    st.deploy()
    n = int(round(cfg["suit"]["airbags"]["deploy_time"] / dt))
    for _ in range(n // 2):
        st.step(model, dt)
    assert 0.3 < airbag_fraction(model, cfg) < 0.7
    for _ in range(n):
        st.step(model, dt)
    assert airbags_deployed(model, cfg)
    st.stow(model)
    assert airbag_fraction(model, cfg) == 0.0 and not st.triggered


def _face_down_drop(model, cfg, deployed):
    """Drop face-down from 0.5 m; returns (airbag geoms that touched the ground, peak torso
    IMU specific force)."""
    set_airbags(model, deployed, cfg)
    ModelRandomizer(model, cfg).apply(1.0, None)            # mj_setConst in either state
    d = mujoco.MjData(model)
    set_initial_pose(model, d, 0.5, np.array([0.7071068, 0.0, 0.7071068, 0.0]), np.zeros(3))
    s = _id(model, mujoco.mjtObj.mjOBJ_SENSOR, imu_accel_sensor("torso"))
    adr = model.sensor_adr[s]
    ground = _id(model, mujoco.mjtObj.mjOBJ_GEOM, GROUND_GEOM)
    bags = set(_id(model, mujoco.mjtObj.mjOBJ_GEOM, airbag_geom(b)) for b in AIRBAGS)
    hit, peak = set(), 0.0
    for _ in range(1000):
        mujoco.mj_step(model, d)
        peak = max(peak, float(np.linalg.norm(d.sensordata[adr:adr + 3])))
        for g1, g2 in d.contact.geom[: d.ncon]:
            if ground in (g1, g2):
                hit |= {int(g1), int(g2)} & bags
    return {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in hit}, peak


def test_airbags_cushion_face_down_drop(model, cfg):
    hit_stowed, peak_stowed = _face_down_drop(model, cfg, False)
    hit_dep, peak_dep = _face_down_drop(model, cfg, True)
    assert hit_stowed == set()                              # stowed bags never touch
    assert airbag_geom("chest") in hit_dep                  # deployed bags make contact
    assert peak_dep < 0.6 * peak_stowed, (peak_dep, peak_stowed)


def test_airbags_do_not_affect_altimeter(model, cfg):
    from suitsim.sensors.suite import Altimeter
    alt = Altimeter(model, cfg["sensors"]["altimeter"])
    d = mujoco.MjData(model)
    set_initial_pose(model, d, 1.0, np.array([0.9238795, 0.0, 0.3826834, 0.0]), np.zeros(3))
    ranges = []
    for dep in (False, True):
        set_airbags(model, dep, cfg)
        mujoco.mj_forward(model, d)
        ranges.append(alt.cast(model, d))
    set_airbags(model, False, cfg)
    assert ranges[0] == pytest.approx(ranges[1])
    assert ranges[0] < alt.max_range


def test_no_self_collision(model):
    d = mujoco.MjData(model)
    set_initial_pose(model, d, 0.5, UP, np.zeros(3))
    set_airbags(model, True)
    mujoco.mj_forward(model, d)
    assert d.ncon == 0   # overlapping limbs/airbags in the air: no contacts at all


@pytest.mark.parametrize("h", [0.0, 1.5, 4.0])
@pytest.mark.parametrize("tilt", [0.0, 0.6, 2.5])
def test_initial_pose_height(model, h, tilt):
    d = mujoco.MjData(model)
    q = np.array([np.cos(tilt / 2), np.sin(tilt / 2) * 0.6, np.sin(tilt / 2) * 0.8, 0.0])
    w = np.array([0.5, -1.0, 2.0])
    set_initial_pose(model, d, h, q, w, linvel=np.array([0.1, 0.0, -0.2]))
    assert lowest_point(model, d) == pytest.approx(h, abs=1e-9)
    root = _id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    np.testing.assert_allclose(d.cvel[root, :3], w, atol=1e-9)   # world-frame ang vel
    np.testing.assert_allclose(d.qvel[:3], [0.1, 0.0, -0.2])
    assert np.all(d.qvel[6:] == 0)


def test_standing(model):
    d = mujoco.MjData(model)
    set_initial_pose(model, d, 0.0, UP, np.zeros(3))
    z0 = d.body("pelvis").xpos[2]
    assert 0.95 < z0 < 1.0
    while d.time < 2.0:
        mujoco.mj_step(model, d)
    assert abs(d.body("pelvis").xpos[2] - z0) < 0.03
    assert d.body("torso").xmat[8] > np.cos(np.deg2rad(20))   # stays upright
    f = sum(d.sensor(foot_touch_sensor(f)).data[0] for f in FEET)
    assert f == pytest.approx(100 * 9.81, rel=0.05)


def test_drop_feet_first(model):
    d = mujoco.MjData(model)
    set_initial_pose(model, d, 3.0, UP, np.zeros(3))
    first, zmin = None, np.inf
    while d.time < 2.0:
        mujoco.mj_step(model, d)
        if first is None and d.ncon:
            first = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                       model.geom_bodyid[d.contact[i].geom[k]])
                     for i in range(d.ncon) for k in (0, 1)} - {"world"}
        zmin = min(zmin, d.body("pelvis").xpos[2])
    assert first and first <= set(FEET)
    assert np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))
    assert zmin > 0.3


def test_free_fall(model):
    d = mujoco.MjData(model)
    set_initial_pose(model, d, 5.0, UP, np.zeros(3))
    for _ in range(200):
        mujoco.mj_step(model, d)
    mujoco.mj_forward(model, d)
    _, v = com_state(model, d)
    assert v[2] == pytest.approx(-9.81 * d.time, rel=1e-3)
    assert np.linalg.norm(d.sensor(imu_accel_sensor("torso")).data) < 1e-3
    mujoco.mj_resetData(model, d)
    set_initial_pose(model, d, 0.0, UP, np.zeros(3))
    for _ in range(500):
        mujoco.mj_step(model, d)
    # standing still: torso accelerometer reads +g (specific force) in body frame
    assert np.linalg.norm(d.sensor(imu_accel_sensor("torso")).data) == pytest.approx(9.81, rel=0.05)


def test_com_state(model):
    d = mujoco.MjData(model)
    set_initial_pose(model, d, 2.0, UP, np.zeros(3), linvel=np.array([1.0, -0.5, 0.3]))
    p, v = com_state(model, d)
    m = model.body_mass
    np.testing.assert_allclose(p, (m[:, None] * d.xipos).sum(0) / m.sum(), atol=1e-9)
    np.testing.assert_allclose(v, [1.0, -0.5, 0.3], atol=1e-9)


def test_randomizer_idempotent(model):
    nom_mass = model.body_mass.copy()
    nom_inertia = model.body_inertia.copy()
    r = ModelRandomizer(model)
    r.apply(1.1, 0.6)
    m1, i1 = model.body_mass.copy(), model.body_inertia.copy()
    r.apply(1.1, 0.6)
    np.testing.assert_allclose(model.body_mass, m1)
    np.testing.assert_allclose(model.body_inertia, i1)
    assert model.body_mass.sum() == pytest.approx(9.0 + 91.0 * 1.1, abs=1e-3)
    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, GROUND_GEOM)
    assert model.geom_friction[g, 0] == pytest.approx(0.6)
    r.restore()
    np.testing.assert_allclose(model.body_mass, nom_mass)
    np.testing.assert_allclose(model.body_inertia, nom_inertia)
    # functional API
    apply_mass_scale(model, 0.9)
    apply_mass_scale(model, 0.9)
    assert model.body_mass.sum() == pytest.approx(9.0 + 91.0 * 0.9, abs=1e-3)
    apply_mass_scale(model, 1.0)
    np.testing.assert_allclose(model.body_mass, nom_mass)
    set_ground_friction(model, 0.5)
    assert model.geom_friction[g, 0] == pytest.approx(0.5)


def test_mjcf_roundtrip(cfg):
    xml = build_mjcf(cfg)
    m = mujoco.MjModel.from_xml_string(xml)
    assert m.nkey == 1 and m.nsite >= 44 + 15 + 3
