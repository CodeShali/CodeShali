"""Sensor-model tests (A3).  Self-contained: a tiny in-test MJCF, not A1's model."""
from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from suitsim.interfaces import FEET, SEGMENTS, SENSOR_KEYS, load_config
from suitsim.sensors import DelayLine, GaussianNoise, SensorSuite, delay_steps


# --------------------------------------------------------------------------------------
# DelayLine / GaussianNoise
# --------------------------------------------------------------------------------------
def test_delay_steps_rounding():
    assert delay_steps(0.002, 0.001) == 2
    assert delay_steps(0.0049999, 0.001) == 5
    assert delay_steps(0.0, 0.001) == 0
    assert delay_steps(0.0014, 0.001) == 1


@pytest.mark.parametrize("n", [0, 1, 2, 5])
def test_delay_line_step_input_exact(n):
    dl = DelayLine(n, (3,))
    dl.fill(np.zeros(3), t=0.0)
    out = []
    for k in range(1, 20):
        dl.push(np.full(3, 1.0 if k >= 10 else 0.0), t=k * 1e-3)
        out.append(dl.get()[0])
        assert dl.get_time() == pytest.approx(max(0, k - n) * 1e-3)
    first = out.index(1.0) + 1          # push index where the step first appears
    assert first == 10 + n


def test_delay_line_unfilled_returns_first_sample():
    dl = DelayLine(3)
    dl.push(7.0)
    assert float(dl.get()) == 7.0
    dl.push(8.0)
    assert float(dl.get()) == 7.0
    with pytest.raises(RuntimeError):
        DelayLine(1).get()


def test_gaussian_noise_statistics():
    rng = np.random.default_rng(0)
    x = np.zeros(200_000)
    assert np.std(GaussianNoise(0.05).apply(x, rng)) == pytest.approx(0.05, rel=0.1)
    assert np.std(GaussianNoise(0.05, scale=2.0).apply(x, rng)) == pytest.approx(0.1, rel=0.1)
    y = np.full(200_000, 4.0)
    rel = GaussianNoise(0.02, relative=True).apply(y, rng)
    assert np.std(rel) == pytest.approx(0.08, rel=0.1)
    assert abs(np.mean(rel) - 4.0) < 0.01
    assert np.array_equal(GaussianNoise(0.0).apply(y, rng), y)


def test_gaussian_noise_seeded():
    a = GaussianNoise(1.0).apply(np.zeros(5), np.random.default_rng(3))
    b = GaussianNoise(1.0).apply(np.zeros(5), np.random.default_rng(3))
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------------------
# Tiny stand-in model: free "pelvis" carrying 15 IMU sites, altimeter, two soles, and a
# hinged "shin" child whose geom lies right under the altimeter ray.
# --------------------------------------------------------------------------------------
def tiny_xml(human_group: int = 0, with_ground: bool = True, shin_angle: float = 0.0) -> str:
    imu_sites = "".join(f'<site name="imu_{s}" pos="0 0 {0.01 * i:.3f}" size="0.005"/>'
                        for i, s in enumerate(SEGMENTS))
    sens = "".join(f'<accelerometer name="acc_{s}" site="imu_{s}"/><gyro name="gyro_{s}" '
                   f'site="imu_{s}"/>' for s in SEGMENTS)
    sens += "".join(f'<touch name="touch_{f}" site="sole_{f}"/>' for f in FEET)
    ground = '<geom name="ground" type="plane" size="20 20 0.1" group="0"/>' if with_ground else ""
    return f"""
<mujoco>
  <option timestep="0.001" gravity="0 0 -9.81"/>
  <worldbody>
    {ground}
    <body name="pelvis" pos="0 0 2">
      <freejoint name="root"/>
      <geom name="pelvis_g" type="box" size="0.15 0.15 0.1" mass="10" group="{human_group}"/>
      {imu_sites}
      <site name="altimeter" pos="0 0 -0.05" size="0.01"/>
      <site name="sole_foot_l" type="box" pos="0.15 0.15 -0.1" size="0.03 0.03 0.02"/>
      <site name="sole_foot_r" type="box" pos="0.15 -0.15 -0.1" size="0.03 0.03 0.02"/>
      <body name="shin" pos="0 0 -0.1">
        <joint name="knee" type="hinge" axis="0 1 0" ref="{shin_angle}"/>
        <geom name="shin_g" type="capsule" fromto="0 0 -0.1 0 0 -0.5" size="0.04" mass="2"
              group="{human_group}" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
  <sensor>{sens}</sensor>
</mujoco>"""


class FakeThrusters:
    def __init__(self, n: int):
        self.forces = np.zeros(n)
        self._rem = {"impulse_gas_pods": 2500.0, "impulse_gas_jets": 600.0, "battery_wh": 150.0}

    def remaining(self):
        return dict(self._rem)


def make(human_group=0, with_ground=True, noise_scale=1.0, seed=0, cfg=None):
    m = mujoco.MjModel.from_xml_string(tiny_xml(human_group, with_ground))
    d = mujoco.MjData(m)
    cfg = cfg or load_config()
    s = SensorSuite(m, cfg, np.random.default_rng(seed))
    s.reset(m, d, noise_scale=noise_scale)
    return m, d, s, cfg


def set_pose(m, d, z=2.0, tilt_deg=0.0, knee=0.0):
    d.qpos[:3] = (0, 0, z)
    a = math.radians(tilt_deg)
    d.qpos[3:7] = (math.cos(a / 2), math.sin(a / 2), 0, 0)
    d.qpos[7] = knee
    mujoco.mj_forward(m, d)


def test_read_keys_and_shapes():
    m, d, s, cfg = make()
    r = s.read()
    assert set(r) == set(SENSOR_KEYS)
    assert r["imu_accel"].shape == (15, 3) and r["imu_gyro"].shape == (15, 3)
    assert r["foot_pressure"].shape == (2,)
    assert r["joint_pos"].shape == (1,) and r["joint_vel"].shape == (1,)
    assert r["thruster_force"].shape == (44,)
    assert set(r["budgets"]) == {"impulse_gas_pods", "impulse_gas_jets", "battery_wh"}


def test_accelerometer_free_fall_vs_rest():
    m, d, s, _ = make(noise_scale=0.0)
    for _ in range(5):
        mujoco.mj_step(m, d)
        s.record(m, d, None)
    assert np.abs(s.read()["imu_accel"]).max() < 1e-6        # free fall: zero specific force
    set_pose(m, d, z=0.1)                                       # box resting on the ground
    for _ in range(400):
        mujoco.mj_step(m, d)
        s.record(m, d, None)
    acc = s.read()["imu_accel"]
    assert np.allclose(acc[:, 2], 9.81, atol=0.05)
    fp = s.read()["foot_pressure"]
    assert fp.sum() > 0   # sole sites touch the ground through the pelvis box contacts


def test_imu_noise_std_and_scale():
    for scale in (1.0, 2.0):
        m, d, s, cfg = make(noise_scale=scale, seed=1)
        set_pose(m, d)
        s.record(m, d, None)
        acc = np.array([s.read()["imu_accel"] for _ in range(3000)])
        gyr = np.array([s.read()["imu_gyro"] for _ in range(3000)])
        assert acc.std(axis=0).mean() == pytest.approx(0.05 * scale, rel=0.1)
        assert gyr.std(axis=0).mean() == pytest.approx(0.01 * scale, rel=0.1)


def test_seeded_reads_reproducible():
    outs = []
    for _ in range(2):
        m, d, s, _ = make(seed=7)
        set_pose(m, d)
        s.record(m, d, None)
        outs.append(s.read()["imu_accel"])
    assert np.array_equal(outs[0], outs[1])


def _run_step_input(sensor_key, apply, delay_s):
    """Apply a step to the true signal at push k0; return push index where it shows up."""
    m, d, s, _ = make(noise_scale=0.0)
    set_pose(m, d)
    s.reset(m, d, noise_scale=0.0)
    k0, seen = 10, None
    base = np.array(s.read()[sensor_key], dtype=float)
    for k in range(1, 30):
        d.time = k * 1e-3
        if k == k0:
            apply(m, d)
        s.record(m, d, None)
        if seen is None and not np.allclose(np.array(s.read()[sensor_key], float), base,
                                            atol=1e-9):
            seen = k
    return seen - k0, delay_s


def test_delay_exactness_per_sensor():
    def knee(m, d):
        d.qpos[7] = 0.3
        mujoco.mj_forward(m, d)

    def spin(m, d):
        d.qvel[3:6] = (0.5, 0, 0)
        mujoco.mj_forward(m, d)

    def lift(m, d):
        d.qpos[2] = 3.0
        mujoco.mj_forward(m, d)

    for key, fn, delay in (("joint_pos", knee, 0.001), ("imu_gyro", spin, 0.002),
                           ("altimeter", lift, 0.005)):
        lag, dly = _run_step_input(key, fn, delay)
        assert lag == round(dly / 1e-3), key


def test_thruster_telemetry_exact_and_undelayed():
    m, d, s, _ = make()
    thr = FakeThrusters(44)
    thr.forces[3] = 123.4
    thr._rem["battery_wh"] = 42.0
    s.record(m, d, thr)
    r = s.read()
    assert r["thruster_force"][3] == 123.4 and r["thruster_force"].sum() == 123.4
    assert r["budgets"]["battery_wh"] == 42.0


@pytest.mark.parametrize("human_group", [0, 1])   # shared group -> multi-pass; split -> mask
def test_altimeter_slant_range_ignores_wearer(human_group):
    m, d, s, _ = make(human_group=human_group, noise_scale=0.0)
    assert s.altimeter.single_pass == (human_group == 1)
    for tilt in (0.0, 20.0, 40.0):
        set_pose(m, d, z=2.0, tilt_deg=tilt)
        r = s.altimeter.cast(m, d)
        site_z = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "altimeter"), 2]
        # the shin capsule lies on the ray but must be ignored -> ground range
        assert r == pytest.approx(site_z / math.cos(math.radians(tilt)), rel=1e-6)
        assert s.altimeter.last_geom == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground")


def test_altimeter_noise_fraction_and_max_range():
    m, d, s, cfg = make(seed=3)
    set_pose(m, d, z=4.0)
    s.reset(m, d)
    true = s.altimeter.cast(m, d)
    rs = np.array([s.read()["altimeter"] for _ in range(5000)])
    assert rs.std() == pytest.approx(0.02 * true, rel=0.1)
    assert abs(rs.mean() - true) < 0.01
    # pointing up / no ground -> max_range, exactly
    set_pose(m, d, z=4.0, tilt_deg=180.0)
    s.reset(m, d)
    assert s.read()["altimeter"] == cfg["sensors"]["altimeter"]["max_range"]
    m2, d2, s2, cfg2 = make(with_ground=False)
    assert s2.read()["altimeter"] == cfg2["sensors"]["altimeter"]["max_range"]


def test_joint_encoder_and_foot_pressure_noise():
    m, d, s, _ = make(seed=5)
    set_pose(m, d, z=0.1)
    for _ in range(300):
        mujoco.mj_step(m, d)
        s.record(m, d, None)
    jp = np.array([s.read()["joint_pos"][0] for _ in range(4000)])
    assert jp.std() == pytest.approx(0.002, rel=0.1)
    fp_true = s._fp.get().copy()
    assert fp_true.min() > 1.0          # both sole sites cover loaded box corners
    k = int(np.argmax(fp_true))
    fp = np.array([s.read()["foot_pressure"][k] for _ in range(4000)])
    assert fp.std() == pytest.approx(0.05 * fp_true[k], rel=0.1)
