"""A tiny stand-in env implementing the suitsim env contract (falling box + 1 thruster).

Used by the eval/viz tests and for smoke-testing ``scripts/evaluate.py`` before the real
envs exist (``--factory suitsim.eval.fake_env:make_fake_env``). Not a physics model of the
suit — just enough MuJoCo to exercise batch evaluation, metrics and rendering.
"""
from __future__ import annotations

import math

from suitsim.viz.gl import ensure_gl

ensure_gl()  # before the first mujoco import

import gymnasium as gym  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from suitsim.interfaces import EPISODE_INFO_KEYS  # noqa: E402

_XML = """
<mujoco model="fake_box">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 .15 .2"
             width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="8 8"/>
  </asset>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" directional="true"/>
    <geom name="ground" type="plane" size="20 20 .1" material="grid"/>
    <body name="pelvis" pos="0 0 2">
      <freejoint name="root"/>
      <geom name="pelvis" type="box" size=".15 .12 .25" pos="0 0 .3" mass="60" rgba=".8 .6 .3 1"/>
      <geom name="foot_l" type="box" size=".14 .11 .03" pos="0 0 0.02" mass="5" rgba=".3 .3 .3 1"/>
      <site name="thr_pod" pos="0 0 .3" zaxis="0 0 1" size=".02"/>
    </body>
  </worldbody>
</mujoco>
"""


class _FakeSpec:
    def __init__(self, name, f_max):
        self.name, self.cls, self.f_max = name, "lift_pod", f_max
        self.site = f"thr_{name}"


class _FakeBank:
    """Mimics the parts of ThrusterBank used for plume drawing."""

    def __init__(self, f_max):
        self.specs = [_FakeSpec("pod", f_max)]
        self.forces = np.zeros(1)


class FakeDropEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, cfg: dict | None = None, controller: str = "baseline",
                 render_mode: str | None = None, log_trajectory: bool = False,
                 frame_skip: int = 5):
        cfg = cfg or {}
        self.cfg = cfg
        task = cfg.get("task") or {}
        self.drop_range = tuple(task.get("drop_height", (1.0, 5.0)))
        self.settle_time = float(task.get("settle_time", 0.3))
        self.max_time = float((cfg.get("sim") or {}).get("episode_time", 3.0))
        safe = cfg.get("safe_landing") or {}
        self.lim_v = float(safe.get("max_vertical_speed", 2.0))
        self.lim_h = float(safe.get("max_horizontal_speed", 1.0))
        self.lim_tilt = float(safe.get("max_tilt_deg", 20.0))
        self.controller = controller
        self.render_mode = render_mode
        self.log_trajectory = log_trajectory
        self.frame_skip = frame_skip
        self.model = mujoco.MjModel.from_xml_string(_XML)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep * frame_skip
        self._bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self._ground = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.mass = float(self.model.body_subtreemass[self._bid])
        self.thrusters = _FakeBank(1500.0)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (6,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (8,), np.float32)
        self._rng = np.random.default_rng(0)
        self._renderer = None

    # ------------------------------------------------------------------ helpers
    def _height(self):
        return float(self.data.xpos[self._bid][2])

    def _tilt_deg(self):
        return math.degrees(math.acos(np.clip(self.data.xmat[self._bid][8], -1, 1)))

    def _obs(self):
        d = self.data
        return np.array([self._height(), *d.qvel[:3], self._tilt_deg() / 90.0,
                         *d.qvel[3:6]], dtype=np.float32)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        rng = self._rng
        mujoco.mj_resetData(self.model, self.data)
        h = rng.uniform(*self.drop_range)
        tilt = rng.uniform(0.0, 30.0)
        axis = rng.uniform(0, 2 * math.pi)
        self.f_max = 1500.0 * rng.uniform(0.45, 1.3)
        self.thrusters.specs[0].f_max = self.f_max
        half = math.radians(tilt) / 2
        self.data.qpos[:3] = [0, 0, h]
        self.data.qpos[3:7] = [math.cos(half), math.sin(half) * math.cos(axis),
                               math.sin(half) * math.sin(axis), 0]
        vh = rng.uniform(-0.5, 0.5, 2)
        self.data.qvel[:2] = vh
        mujoco.mj_forward(self.model, self.data)
        self.params = {"drop_height": h, "tilt_deg": tilt, "thrust_scale": self.f_max / 1500.0,
                       "vh0": float(np.linalg.norm(vh))}
        self.impulse = 0.0
        self.battery = 0.0
        self.contact_t = None
        self.impact = None
        self.first_contact = "none"
        self.traj = []
        self.thrusters.forces[:] = 0
        return self._obs(), {}

    def _baseline_thrust(self):
        vz = self.data.qvel[2]
        h = self._height()
        # constant-deceleration braking toward -1.0 m/s at the ground
        v_ref = -math.sqrt(1.0 + 2 * 6.0 * max(h, 0.0))
        a = 9.81 + 8.0 * (v_ref - vz)
        return float(np.clip(self.mass * a, 0.0, self.f_max))

    def step(self, action):
        action = np.asarray(action, dtype=float).reshape(-1)
        f = 0.0
        if self.contact_t is None and self.controller in ("baseline", "residual"):
            f = self._baseline_thrust()
            if self.controller == "residual":
                f = float(np.clip(f + 0.3 * self.f_max * action[2], 0.0, self.f_max))
        self.thrusters.forces[0] = f
        for _ in range(self.frame_skip):
            # mostly vertical push (fake attitude control), slightly along the body axis
            zax = self.data.xmat[self._bid].reshape(3, 3)[:, 2]
            self.data.xfrc_applied[self._bid, :3] = f * (0.8 * np.array([0, 0, 1.0]) + 0.2 * zax)
            mujoco.mj_step(self.model, self.data)
            self.impulse += f * self.model.opt.timestep
            self.battery += 0.001 * f * self.model.opt.timestep
            if self.contact_t is None:
                for c in self.data.contact[: self.data.ncon]:
                    if self._ground in (c.geom1, c.geom2):
                        other = c.geom2 if c.geom1 == self._ground else c.geom1
                        self.first_contact = mujoco.mj_id2name(
                            self.model, mujoco.mjtObj.mjOBJ_GEOM, other)
                        v = self.data.qvel[:3]
                        self.impact = (float(-v[2]), float(np.linalg.norm(v[:2])),
                                       self._tilt_deg())
                        self.contact_t = float(self.data.time)
                        break
        t = float(self.data.time)
        if self.log_trajectory:
            self.traj.append({"t": t, "com_height": self._height(),
                              "com_vel": self.data.qvel[:3].copy().tolist(),
                              "tilt": math.radians(self._tilt_deg()),
                              "forces_by_class": {"lift_pod": f, "fan": 0.0, "micro_jet": 0.0},
                              "budgets": {"impulse_gas_pods": 2500.0 - self.impulse,
                                          "impulse_gas_jets": 600.0,
                                          "battery_wh": 150.0 - self.battery},
                              "phase": "brake" if f > 0 else "idle"})
        terminated = self.contact_t is not None and t - self.contact_t >= self.settle_time
        truncated = (not terminated) and t >= self.max_time
        info = {}
        reward = -0.001 * f * self.dt
        if terminated or truncated:
            info = self._episode_info(t)
            reward += 10.0 * info["safe"]
        return self._obs(), reward, terminated, truncated, info

    def _episode_info(self, t):
        vv, vh, tilt = self.impact if self.impact else (0.0, 0.0, self._tilt_deg())
        bad = self.first_contact == "pelvis"
        landed = self.first_contact != "none"
        safe = landed and not bad and vv <= self.lim_v and vh <= self.lim_h \
            and tilt <= self.lim_tilt
        info = {"safe": bool(safe), "impact_v_vertical": vv, "impact_v_horizontal": vh,
                "tilt_at_impact_deg": tilt, "first_contact": self.first_contact,
                "bad_contact": bool(bad), "impulse_gas_pods": self.impulse,
                "impulse_gas_jets": 0.0, "battery_wh": self.battery,
                "airbags_deployed": False, "success": bool(safe), "episode_time": t,
                "episode_params": dict(self.params)}
        assert set(EPISODE_INFO_KEYS) <= set(info)
        if self.log_trajectory:
            info["trajectory"] = list(self.traj)
        return info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 240, 320)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def make_fake_env(cfg: dict | None = None, controller: str = "baseline", **kw) -> FakeDropEnv:
    """Factory with the same call shape as ``suitsim.envs.make_env`` minus the task name."""
    return FakeDropEnv(cfg, controller=controller, **kw)
