"""Shared simulation core for every suit task env (T1 drop, T2 jump, T3 push).

:class:`SuitEnvBase` owns the MuJoCo model, the thruster bank, sensors, estimator, allocator,
domain randomiser and the controller, and runs the nested loop

* physics step (``sim.timestep``, 1 ms): ``thrusters.step(u)`` -> per-body thruster wrench
  (+ whole-body drag at the root CoM) -> ``data.xfrc_applied`` -> ``mj_step`` ->
  ``sensors.record`` -> ground-contact bookkeeping (ground truth, metrics only);
* control step (``sim.control_dt``, 2 ms): ``est = estimator.update(sensors.read())`` ->
  ``ctrl = controller.compute(est)`` (+ residual wrench) ->
  ``u = allocator.allocate(wrench, *estimator.thruster_geometry(est), thrusters.u_max)``
  -> airbags;
* policy / env step (``policy_dt``, 20 ms = 10 control steps).

Subclasses implement :meth:`_sample_episode` (initial state + DR draw), :meth:`_reward`,
:meth:`_done` and :meth:`_task_info`.

Notes
-----
* Thruster forces are applied with the kinematics of the previous physics step
  (``mj_step`` leaves ``site_xpos`` at the pre-integration state) — a 1 ms lag, negligible
  next to the 5 ms thruster time constant, and it avoids an extra ``mj_kinematics`` per step.
* Drag is recomputed once per control step from the ground-truth CoM velocity (it is part
  of the plant, not of the controller) and applied at the pelvis CoM (zero torque).
* Ground-truth impact quantities are read at the physics step whose contact list first
  holds a ground contact: ``data.contact`` and ``cvel`` both refer to the pre-integration
  state of that step, i.e. the velocity just before the contact impulse acts.  The impact
  velocity is the whole-body CoM velocity (``mj_subtreeVel`` of the root subtree).
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any

# Headless rendering: MuJoCo picks its GL backend when first imported; osmesa is the reliable
# software backend on the dev container. A user-set MUJOCO_GL wins.  If mujoco was imported
# before this module without MUJOCO_GL, the GLFW default is locked in and offscreen rendering
# would abort the process -> render() raises instead (import suitsim.envs / call
# suitsim.viz.gl.ensure_gl() before mujoco).
_GL_LOCKED_OUT = "mujoco" in sys.modules and not os.environ.get("MUJOCO_GL")
os.environ.setdefault("MUJOCO_GL", "osmesa")
# osmesa (llvmpipe) and triton bundle clashing LLVM symbols: importing triton AFTER mujoco has
# loaded libOSMesa segfaults (this happens when SB3/torch builds an optimizer -> torch._dynamo
# -> triton, e.g. loading a residual policy in an eval worker). Loading triton first is safe
# (~0.1 s); skipped when triton is absent or mujoco is already imported.
if os.environ.get("MUJOCO_GL") == "osmesa" and "mujoco" not in sys.modules:
    try:
        import triton  # noqa: F401
    except Exception:
        pass

import gymnasium as gym  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from suitsim.actuators import ThrusterBank, apply_thrusters, body_drag_wrench, thruster_ids
from suitsim.actuators.propellant import PropellantTracker
from suitsim.control.allocation import Allocator
from suitsim.control.fast_allocation import FastAllocator
from suitsim.control.baseline import BaselineLandingController
from suitsim.estimation import StateEstimator
from suitsim.interfaces import (
    BUDGET_KEYS, FEET, G0, GROUND_GEOM, HEAD_TORSO_BODIES, ROOT_BODY, THRUSTER_CLASSES,
    TORSO_BODY, ControlOutput, load_thruster_specs, propellant_specs, propellant_tracking,
)
from suitsim.model import AirbagState, ModelRandomizer, build_model, model_info
from suitsim.sensors import SensorSuite

CONTROLLERS = ("baseline", "residual", "none")

_OBS_DEFAULTS = {
    "height_scale": 5.0,        # [m]
    "vel_scale": 10.0,          # [m/s]
    "angvel_scale": 5.0,        # [rad/s]
    "time_scale": 5.0,          # [s]
    "wrench_scale": [3000.0, 3000.0, 3000.0, 300.0, 300.0, 100.0],
    "clip": 10.0,
}


class SuitEnvBase(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, cfg: dict, controller: str = "baseline", render_mode: str | None = None,
                 log_trajectory: bool = False, policy_dt: float | None = None,
                 render_size: tuple[int, int] = (480, 640)):
        if controller not in CONTROLLERS:
            raise ValueError(f"controller must be one of {CONTROLLERS}, got {controller!r}")
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        self.cfg = cfg
        self.controller_name = controller
        self.render_mode = render_mode
        self.log_trajectory = bool(log_trajectory)
        self.render_size = tuple(render_size)

        sim = cfg["sim"]
        self.model = build_model(cfg)
        self.data = mujoco.MjData(self.model)
        self.dt_phys = float(self.model.opt.timestep)
        self.control_dt = float(sim["control_dt"])
        self.policy_dt = float(policy_dt if policy_dt is not None else cfg.get("policy_dt", 0.02))
        self.n_phys = max(1, int(round(self.control_dt / self.dt_phys)))
        self.n_ctrl = max(1, int(round(self.policy_dt / self.control_dt)))
        self.episode_time = float((cfg.get("task") or {}).get("episode_time",
                                                               sim.get("episode_time", 5.0)))
        self.g = float(sim.get("gravity", 9.81))
        self.metadata = dict(self.metadata, render_fps=int(round(1.0 / self.policy_dt)))

        self.info_m = model_info(self.model, cfg)
        self.specs = load_thruster_specs(cfg)
        self.thrusters = ThrusterBank(self.specs, cfg, self.dt_phys)
        self.site_ids, self.body_ids = thruster_ids(self.model, self.specs)
        self.sensors = SensorSuite(self.model, cfg, np.random.default_rng(0))
        self.estimator = StateEstimator(self.model, cfg)
        solver = str((cfg.get("allocation", {}) or {}).get("solver", "active_set"))
        self.allocator = (FastAllocator if solver == "active_set" else Allocator)(self.specs, cfg)
        self.randomizer = ModelRandomizer(self.model, cfg)
        # propellant mass (suit.propellant): reported always, depleted in the model only with
        # suit.propellant_mass_tracking (composes with the randomiser, see _apply_dr_common)
        self.prop_specs = propellant_specs(cfg)
        self.propellant = (PropellantTracker.from_config(self.model, cfg)
                           if self.prop_specs and propellant_tracking(cfg) else None)
        self.airbag_state = AirbagState(self.model, cfg)
        self.controller = None if controller == "none" else BaselineLandingController(
            cfg, self.specs, geometry_fn=self.estimator.thruster_geometry)

        m = self.model
        self.root_id = self.info_m["root_body_id"]
        self.torso_id = self.info_m["body_ids"][TORSO_BODY]
        self.ground_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, GROUND_GEOM)
        self.body_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
        self.feet_ids = {self.info_m["body_ids"][f] for f in FEET}
        self.bad_ids = {self.info_m["body_ids"][b] for b in HEAD_TORSO_BODIES}
        self.cls_idx = {c: self.thrusters.class_index(c) for c in THRUSTER_CLASSES}
        self.sole_sites = self.info_m["sole_site_ids"]
        self.env_geoms = self._environment_geoms()

        res = cfg.get("residual", {}) or {}
        self.residual_scale = np.concatenate([
            np.asarray(res.get("max_force", [0, 0, 0]), float),
            np.asarray(res.get("max_torque", [0, 0, 0]), float)])
        oc = dict(_OBS_DEFAULTS)
        oc.update(cfg.get("observation", {}) or {})
        self.obs_cfg = oc
        self.wrench_scale = np.asarray(oc["wrench_scale"], float)
        nj = len(self.info_m["joint_names"])
        self.obs_dim = 1 + 1 + 3 + 4 + 3 + nj + len(BUDGET_KEYS) + 1 + 1 + len(FEET) + 6
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (6,), np.float32)
        self.capacity = np.array([self.thrusters.capacity[k] for k in BUDGET_KEYS])

        self._renderer = None
        self._xfrc = np.zeros((m.nbody, 6))
        self.params: dict[str, Any] = {}
        self._episode_over = True

    # ------------------------------------------------------------------ subclass hooks
    def _environment_geoms(self) -> set[int]:
        """Geoms that count as 'ground' for contact bookkeeping (world-body geoms)."""
        m = self.model
        return {g for g in range(m.ngeom) if m.geom_bodyid[g] == 0}

    def _sample_episode(self, rng: np.random.Generator) -> dict:
        """Draw DR + initial state; must set the MuJoCo state. Returns episode params."""
        raise NotImplementedError

    def _reward(self, terminal: bool, info: dict) -> float:
        return 0.0

    def _done(self) -> bool:
        return False

    def _task_info(self) -> dict:
        return {}

    # ------------------------------------------------------------------ helpers
    def _truth_com(self) -> tuple[np.ndarray, np.ndarray]:
        mujoco.mj_subtreeVel(self.model, self.data)
        return (self.data.subtree_com[self.root_id].copy(),
                self.data.subtree_linvel[self.root_id].copy())

    def _truth_tilt(self) -> float:
        return float(math.acos(max(-1.0, min(1.0, self.data.xmat[self.torso_id][8]))))

    def _apply_dr_common(self, rng: np.random.Generator) -> dict:
        """Mass, friction, thruster f_max / tau scales, sensor noise (spec 5.4)."""
        rc = self.cfg.get("randomization", {}) or {}
        fixed = (self.cfg.get("task") or {}).get("fixed") or {}
        on = bool(rc.get("enabled", True))
        n = self.thrusters.n
        mass_scale = 1.0 + rng.uniform(-1, 1) * float(rc.get("human_mass_frac", 0.0))
        lo, hi = rc.get("ground_friction", [1.0, 1.0])
        friction = rng.uniform(float(lo), float(hi))
        sf = 1.0 + rng.uniform(-1, 1, n) * float(rc.get("thruster_force_frac", 0.0))
        st = 1.0 + rng.uniform(-1, 1, n) * float(rc.get("response_delay_frac", 0.0))
        lo, hi = rc.get("sensor_noise_scale", [1.0, 1.0])
        noise = rng.uniform(float(lo), float(hi))
        if not on:
            mass_scale, noise = 1.0, 1.0
            friction = float(self.cfg["sim"].get("ground_friction", 1.0))
            sf, st = np.ones(n), np.ones(n)
        mass_scale = float(fixed.get("mass_scale", mass_scale))
        friction = float(fixed.get("friction", friction))
        noise = float(fixed.get("noise_scale", noise))
        if "fmax_scale" in fixed:
            sf = np.full(n, float(fixed["fmax_scale"]))
        if "tau_scale" in fixed:
            st = np.full(n, float(fixed["tau_scale"]))
        self.randomizer.apply(mass_scale, friction)     # full tanks (nominal suit masses)
        if self.propellant is not None:
            self.propellant.capture()                    # this episode's full-load state
        self.thrusters.reset(scale_fmax=sf, scale_tau=st)
        self._noise_scale = noise
        return {"mass_scale": mass_scale, "friction": friction, "noise_scale": noise,
                "fmax_scale_mean": float(sf.mean()), "fmax_scale_min": float(sf.min()),
                "pod_fmax_scale_mean": float(sf[self.cls_idx["lift_pod"]].mean()),
                "tau_scale_mean": float(st.mean()),
                "total_mass": float(self.model.body_subtreemass[self.root_id])}

    def _obs(self, est: dict, wrench: np.ndarray) -> np.ndarray:
        oc = self.obs_cfg
        bud = np.array([est["budgets"][k] for k in BUDGET_KEYS]) / self.capacity
        o = np.concatenate([
            [est["com_height"] / oc["height_scale"], est["feet_height"] / oc["height_scale"]],
            np.asarray(est["com_vel"]) / oc["vel_scale"],
            est["torso_quat"],
            np.asarray(est["torso_angvel"]) / oc["angvel_scale"],
            est["joint_pos"],
            bud,
            [est["time_since_fall"] / oc["time_scale"], float(est["free_fall"])],
            np.asarray(est["foot_contact"], float),
            np.asarray(wrench) / self.wrench_scale,
        ])
        c = float(oc["clip"])
        return np.clip(np.nan_to_num(o), -c, c).astype(np.float32)

    # ------------------------------------------------------------------ gym API
    @property
    def dt(self) -> float:
        """Wall-clock duration of one env step (one policy step) [s]."""
        return self.n_ctrl * self.n_phys * self.dt_phys

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        ss = np.random.SeedSequence(int(self.np_random.integers(2 ** 63)))
        r_task, r_sens, r_est = (np.random.default_rng(s) for s in ss.spawn(3))
        mujoco.mj_resetData(self.model, self.data)
        if self.propellant is not None:
            self.propellant.restore()                    # refill (before the DR draw)
        self.airbag_state.stow(self.model)
        self.params = self._sample_episode(r_task)
        d = self.data
        self.sensors.rng = r_sens
        self.sensors.reset(self.model, d, self._noise_scale)
        com, vel = self._truth_com()
        hint = {"torso_quat": d.xquat[self.torso_id].copy(), "com_vel": vel,
                "com_height": float(com[2]), "com_pos": com}
        self.estimator.reset(hint, r_est)
        if self.controller is not None:
            self.controller.reset()
        if hasattr(self.allocator, "reset_warm_start"):
            self.allocator.reset_warm_start()     # per-seed determinism
        self.est = self.estimator.update(self.sensors.read(), 0.0)
        self.u = np.zeros(self.thrusters.n)
        self.wrench = np.zeros(6)
        self.base_wrench = np.zeros(6)
        self.phase = "idle"
        self.ctrl_debug = {}
        self.airbags_deployed = False
        self.airbag_time = None
        # contact bookkeeping (ground truth)
        self.first_contact = "none"
        self.t_first_contact = None
        self.t_first_foot = None
        self.bad_contact = False
        self.any_bad_contact = False
        self.impact = None
        self.n_alloc = 0
        self.alloc_residual_sum = 0.0
        self.traj: list[dict] = []
        self._episode_over = False
        self._drag = np.zeros(3)
        return self._obs(self.est, self.base_wrench), {"episode_params": dict(self.params)}

    def _control_step(self, action: np.ndarray | None) -> None:
        est = self.estimator.update(self.sensors.read(), self.control_dt)
        self.est = est
        if self.controller is None:
            ctrl = ControlOutput(wrench=np.zeros(6), phase="none")
        else:
            ctrl = self.controller.compute(est, None)
        self.phase = ctrl.phase
        self.ctrl_debug = ctrl.debug
        self.base_wrench = np.asarray(ctrl.wrench, float)
        w = self.base_wrench
        if (self.controller_name == "residual" and action is not None
                and ctrl.phase not in ("idle", "landed")):
            w = w + np.clip(action, -1.0, 1.0) * self.residual_scale
        self.wrench = w
        if ctrl.thrust_cmd is not None:
            self.u = np.asarray(ctrl.thrust_cmd, float)
        elif np.any(w):
            r, dvec = self.estimator.thruster_geometry(est)
            self.u = self.allocator.allocate(w, r, dvec, self.thrusters.u_max)
            self.n_alloc += 1
            self.alloc_residual_sum += self.allocator.last_residual
        else:
            self.u = np.zeros(self.thrusters.n)
        if ctrl.deploy_airbags and not self.airbags_deployed:
            self.airbag_state.deploy()            # inflates over suit.airbags.deploy_time
            self.airbags_deployed = True
            self.airbag_time = float(self.data.time)
        _, vel = self._truth_com()
        self._drag = body_drag_wrench(self.model, self.data, self.cfg, self.root_id, vel)

    def _physics_step(self) -> None:
        m, d = self.model, self.data
        self.airbag_state.step(m, self.dt_phys)
        f = self.thrusters.step(self.u)
        if self.propellant is not None:
            self.propellant.update(self.thrusters.used())   # mass for the coming mj_step
        apply_thrusters(m, d, self.site_ids, self.body_ids, f, out=self._xfrc)
        self._xfrc[self.root_id, :3] += self._drag
        d.xfrc_applied[:] = self._xfrc
        mujoco.mj_step(m, d)
        self.sensors.record(m, d, self.thrusters)
        if d.ncon:
            self._contacts()

    def _contacts(self) -> None:
        d, m = self.data, self.model
        geoms = d.contact.geom[: d.ncon]
        bodies = set()
        for g1, g2 in geoms:
            if g1 in self.env_geoms and g2 not in self.env_geoms:
                bodies.add(int(m.geom_bodyid[g2]))
            elif g2 in self.env_geoms and g1 not in self.env_geoms:
                bodies.add(int(m.geom_bodyid[g1]))
        if not bodies:
            return
        t = float(d.time) - self.dt_phys        # contact list refers to the pre-step state
        feet = bodies & self.feet_ids
        bad = bodies & self.bad_ids
        if bad:
            self.any_bad_contact = True
        if self.t_first_contact is None:
            self.t_first_contact = t
            b = min(feet) if feet else (min(bad) if bad else min(bodies))
            self.first_contact = self.body_names[b]
        if bad and self.t_first_foot is None and not feet:
            self.bad_contact = True
        if (feet and self.t_first_foot is None) or (self.impact is None and self.t_first_contact == t):
            # impact state from the pre-step kinematics (cvel/xmat of the forward pass)
            mujoco.mj_subtreeVel(m, d)
            vel = d.subtree_linvel[self.root_id].copy()
            self.impact = {"v": vel, "tilt": self._truth_tilt(), "t": t,
                           "body": self.body_names[min(feet) if feet else min(bodies)]}
            if feet:
                self.t_first_foot = t

    def _log(self) -> None:
        com, vel = self._truth_com()
        f = self.thrusters.forces
        est = self.est
        soles = self.data.site_xpos[self.sole_sites][:, 2]
        self.traj.append({
            "t": float(self.data.time),
            "com_height": float(com[2]), "feet_height": float(soles.min()),
            "com_vel": vel.tolist(), "tilt": self._truth_tilt(),
            "forces_by_class": {c: float(f[i].sum()) for c, i in self.cls_idx.items()},
            "budgets": self.thrusters.remaining(),
            "phase": self.phase,
            "wrench_cmd": self.wrench.tolist(),
            "alloc_residual": float(self.allocator.last_residual),
            "est_com_height": float(est["com_height"]), "est_feet_height": float(est["feet_height"]),
            "est_com_vel": np.asarray(est["com_vel"]).tolist(), "est_tilt": float(est["tilt"]),
            "est_free_fall": bool(est["free_fall"]),
            "airbags": self.airbags_deployed,
            "ctrl": dict(self.ctrl_debug),
        })

    def step(self, action):
        if self._episode_over:
            raise RuntimeError("step() called after the episode ended; call reset()")
        action = None if action is None else np.asarray(action, dtype=float).reshape(6)
        terminated = truncated = False
        for _ in range(self.n_ctrl):
            self._control_step(action)
            for _ in range(self.n_phys):
                self._physics_step()
            if self.log_trajectory:
                self._log()
            terminated = self._done()
            truncated = (not terminated) and self.data.time >= self.episode_time - 1e-9
            if terminated or truncated:
                break
        info: dict = {"phase": self.phase}
        end = terminated or truncated
        if end:
            info.update(self._episode_info())
            self._episode_over = True
        reward = self._reward(end, info)
        return self._obs(self.est, self.base_wrench), float(reward), terminated, truncated, info

    def _episode_info(self) -> dict:
        used = self.thrusters.used()
        info = {
            "impulse_gas_pods": used["impulse_gas_pods"],
            "impulse_gas_jets": used["impulse_gas_jets"],
            "battery_wh": used["battery_wh"],
            "airbags_deployed": bool(self.airbags_deployed),
            "episode_time": float(self.data.time),
            "episode_params": dict(self.params),
            "alloc_calls": self.n_alloc,
            "alloc_residual_mean": self.alloc_residual_sum / max(self.n_alloc, 1),
        }
        info.update(self._propulsion_info(used))
        info.update(self._task_info())
        if self.log_trajectory:
            info["trajectory"] = list(self.traj)
        return info

    def _propulsion_info(self, used: dict) -> dict:
        """Extra info of the hardware-realistic propulsion options (empty for the default
        suit, so its results files are unchanged)."""
        out: dict = {}
        if self.prop_specs:
            tot = 0.0
            for ps in self.prop_specs:
                kg = float(used[ps.budget]) / (ps.isp * G0)
                out[f"propellant_kg_{ps.budget.replace('impulse_gas_', '')}"] = kg
                tot += kg
            out["propellant_kg"] = tot
            out["propellant_mass_tracked"] = self.propellant is not None
            out["total_mass_end"] = float(self.model.body_subtreemass[self.root_id])
        if self.thrusters.idx_pulse.size:
            out["pulse_charges_used"] = self.thrusters.charges_used()
            log = self.thrusters.ignition_log()
            out["pulse_first_ignition_t"] = float(log[0][0]) if log else float("nan")
        if self.thrusters.idx_valve.size:
            out["valve_pulses"] = self.thrusters.valve_pulses()
        return out

    # ------------------------------------------------------------------ rendering
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            if _GL_LOCKED_OUT:
                raise RuntimeError("mujoco was imported before suitsim.envs without MUJOCO_GL; "
                                   "set MUJOCO_GL=osmesa|egl (or call "
                                   "suitsim.viz.gl.ensure_gl()) before importing mujoco")
            h, w = self.render_size
            self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, w)
            self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, h)
            self._renderer = mujoco.Renderer(self.model, h, w)
        self._renderer.update_scene(self.data, camera="track")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
