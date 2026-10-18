"""T3 — push recovery on a 0.3 m wide beam (milestone M6, part 2).

The wearer stands on a static box geom ``beam`` (top ``beam_height`` above the ground,
axis along world x, the wearer faces +x).  At a random time in ``push_time`` a horizontal
force of ``push_force`` N in a uniformly random azimuth acts for ``push_duration`` s on the
torso CoM (ground-truth disturbance; the controller only sees its effect through the
sensors).  Pass (``success``): at the episode end both feet are on the beam top, the pelvis
is up, the torso is not tilted beyond ``push_success.max_tilt_deg`` and no body part other
than the feet ever touched the beam or the ground.

Implementation notes
--------------------
* The beam is injected into the generated MJCF before compilation (``_build_beam_model``);
  :class:`SuitEnvBase` builds its model through ``suitsim.envs.base.build_model``, which is
  swapped for the duration of ``super().__init__`` (no edit of ``base.py`` needed).  Beam
  contact bits are those of the ground (contype 2 / conaffinity 1, priority 1), so it is a
  world-body geom and counts as "environment" for the base contact bookkeeping.
* The push is written into ``data.qfrc_applied`` with ``mj_applyFT`` every physics step of
  the push window (``xfrc_applied`` is owned by the base thruster loop).
* ``controller='baseline'`` maps to :class:`suitsim.control.balance.BalanceController`.
  The controller may also return posture-servo offsets (``ControlOutput.debug
  ['posture_offset']``, rad, hinge order), applied here to ``data.ctrl``.
* Info keys of the landing tasks are filled sensibly: ``first_contact`` is the first foot on
  the beam at t=0, impact quantities are those of that (zero-speed) initial contact,
  ``safe == success``.
"""
from __future__ import annotations

import math
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

import suitsim.envs.base as _base
from suitsim.envs.base import SuitEnvBase
from suitsim.interfaces import FEET, ROOT_BODY, TORSO_BODY
from suitsim.model import build_mjcf, set_initial_pose

BEAM_GEOM = "beam"


def _build_beam_model(cfg: dict) -> mujoco.MjModel:
    """Human + suit model with the static beam added to the world body."""
    task = cfg["task"]
    w, top, length = (float(task["beam_width"]), float(task["beam_height"]),
                      float(task["beam_length"]))
    c = cfg["human"]["contact"]
    mu = float(cfg["sim"]["ground_friction"])
    xml = build_mjcf(cfg)
    header, _, body = xml.partition("-->\n") if xml.startswith("<!--") else ("", "", xml)
    root = ET.fromstring(body)
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name=BEAM_GEOM, type="box",
                  pos=f"0 0 {top / 2:.6g}", size=f"{length / 2:.6g} {w / 2:.6g} {top / 2:.6g}",
                  rgba="0.55 0.42 0.28 1", friction=f"{mu:.6g} 0.005 0.0001",
                  contype=str(c["ground_contype"]), conaffinity=str(c["ground_conaffinity"]),
                  priority=str(c["ground_priority"]), condim=str(c["condim"]))
    # the generated 'stand' keyframe refers to the ground; drop it (poses are set explicitly)
    kf = root.find("keyframe")
    if kf is not None:
        root.remove(kf)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


class PushRecoveryEnv(SuitEnvBase):
    """Push recovery on a narrow beam. See module docstring."""

    def __init__(self, cfg: dict, controller: str = "baseline", render_mode: str | None = None,
                 log_trajectory: bool = False, policy_dt: float | None = None, **kw):
        orig = _base.build_model
        _base.build_model = _build_beam_model
        try:
            super().__init__(cfg, controller=controller, render_mode=render_mode,
                             log_trajectory=log_trajectory, policy_dt=policy_dt, **kw)
        finally:
            _base.build_model = orig
        task = cfg["task"]
        self.beam_width = float(task["beam_width"])
        self.beam_top = float(task["beam_height"])
        self.beam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, BEAM_GEOM)
        self.push_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                              str(task.get("push_body", TORSO_BODY)))
        self.ps = dict(cfg.get("push_success", {}) or {})
        self.rw = dict(cfg.get("reward", {}) or {})
        self.servo_ids = np.asarray(self.info_m["servo_ids"], int)
        self.posture0 = np.asarray(self.info_m["nominal_posture"], float)
        self.foot_body_ids = [self.info_m["body_ids"][f] for f in FEET]
        self.foot_geom_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                                f"geom_{f}") for f in FEET]
        # cache the latest sensor reading so the controller can use raw sensors as well
        self._last_sensors: dict | None = None
        _read = self.sensors.read

        def _read_cached():
            s = _read()
            self._last_sensors = s
            return s
        self.sensors.read = _read_cached
        if controller != "none":
            from suitsim.control.balance import BalanceController
            self.controller = BalanceController(
                cfg, self.specs, estimator=self.estimator,
                sensors_fn=lambda: self._last_sensors)
        self._push_F = np.zeros(3)
        self._push_t0 = math.inf
        self._push_t1 = math.inf
        self._push_active = False

    # ------------------------------------------------------------------ episode
    def _sample_episode(self, rng: np.random.Generator) -> dict:
        task = self.cfg["task"]
        fixed = task.get("fixed") or {}
        pt = task.get("push_time", [0.5, 1.5])
        t_push = rng.uniform(*pt) if isinstance(pt, (list, tuple)) else float(pt)
        pd = task.get("push_direction_deg", [0.0, 360.0])
        az = rng.uniform(*pd) if isinstance(pd, (list, tuple)) else float(pd)
        sway = float(task.get("initial_sway", 0.0))
        sway_dir = rng.uniform(0, 2 * math.pi)
        sway_v = sway * math.sqrt(rng.uniform())
        dr = self._apply_dr_common(rng)
        t_push = float(fixed.get("push_time", t_push))
        az = float(fixed.get("push_direction_deg", az))
        force = float(fixed.get("push_force", task.get("push_force", 200.0)))
        dur = float(fixed.get("push_duration", task.get("push_duration", 0.2)))
        # beam friction follows the randomised ground friction
        self.model.geom_friction[self.beam_id] = self.model.geom_friction[self.ground_id]

        v0 = sway_v * np.array([math.cos(sway_dir), math.sin(sway_dir), 0.0])
        set_initial_pose(self.model, self.data, self.beam_top, np.array([1.0, 0, 0, 0]),
                         np.zeros(3), linvel=v0, cfg=self.cfg)
        a = math.radians(az)
        self._push_F = force * np.array([math.cos(a), math.sin(a), 0.0])
        self._push_t0, self._push_t1 = t_push, t_push + dur
        self._push_active = False
        self.non_foot_contact = False
        self.non_foot_body = "none"
        self.t_fall = None
        lat = math.degrees(math.atan2(abs(math.sin(a)), abs(math.cos(a))))
        return {"push_direction_deg": az % 360.0, "push_time": t_push, "push_force": force,
                "push_duration": dur, "push_impulse": force * dur,
                "push_lateral_deg": lat,               # 0 = along the beam, 90 = across
                "push_fx": float(self._push_F[0]), "push_fy": float(self._push_F[1]),
                "initial_sway": float(sway_v), **dr}

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # the estimator's height channel is referenced to the ground (z=0); the altimeter
        # sees the beam top -> reference the initial height to the surface below
        self.estimator.x[0] -= self.beam_top
        self.est["com_height"] -= self.beam_top
        self.est["feet_height"] -= self.beam_top
        return self._obs(self.est, self.base_wrench), info

    # ------------------------------------------------------------------ dynamics
    def _physics_step(self) -> None:
        d = self.data
        t = float(d.time)
        on = self._push_t0 - 1e-9 <= t < self._push_t1 - 1e-9
        if on:
            d.qfrc_applied[:] = 0.0
            mujoco.mj_applyFT(self.model, d, self._push_F, np.zeros(3),
                              d.xipos[self.push_body_id], self.push_body_id, d.qfrc_applied)
        elif self._push_active:
            d.qfrc_applied[:] = 0.0
        self._push_active = on
        super()._physics_step()
        if d.ncon:
            self._non_foot_contacts()

    def _non_foot_contacts(self) -> None:
        d, m = self.data, self.model
        env = self.env_geoms
        for g1, g2 in d.contact.geom[: d.ncon]:
            if g1 in env and g2 not in env:
                b = int(m.geom_bodyid[g2])
            elif g2 in env and g1 not in env:
                b = int(m.geom_bodyid[g1])
            else:
                continue
            if b not in self.feet_ids:
                if not self.non_foot_contact:
                    self.non_foot_contact = True
                    self.non_foot_body = self.body_names[b]
                    if self.t_fall is None:
                        self.t_fall = float(d.time)
                return

    def _control_step(self, action) -> None:
        super()._control_step(action)
        off = self.ctrl_debug.get("posture_offset") if self.ctrl_debug else None
        if off is not None:
            self.data.ctrl[self.servo_ids] = self.posture0 + np.asarray(off, float)

    # ------------------------------------------------------------------ outcome
    def _pelvis_height(self) -> float:
        return float(self.data.xpos[self.root_id][2] - self.beam_top)

    def _done(self) -> bool:
        if self.non_foot_contact:
            return True
        if self._pelvis_height() < float(self.ps.get("fall_pelvis_height", 0.45)):
            if self.t_fall is None:
                self.t_fall = float(self.data.time)
            return True
        return False

    def _feet_state(self) -> tuple[bool, list[str]]:
        d = self.data
        tol = float(self.ps.get("sole_height_tol", 0.04))
        lat = self.beam_width / 2 + float(self.ps.get("foot_lateral_margin", 0.0))
        soles = d.site_xpos[self.sole_sites]
        reasons = []
        for k, f in enumerate(FEET):
            dz = soles[k, 2] - self.beam_top
            yc = float(d.geom_xpos[self.foot_geom_ids[k]][1])
            if not (-tol <= dz <= tol):
                reasons.append(f"{f}_off_height")
            if abs(yc) > lat:
                reasons.append(f"{f}_off_side")
        return not reasons, reasons

    def _task_info(self) -> dict:
        feet_ok, reasons = self._feet_state()
        tilt = math.degrees(self._truth_tilt())
        ph = self._pelvis_height()
        if self.non_foot_contact:
            reasons.insert(0, f"contact_{self.non_foot_body}")
        if ph < float(self.ps.get("min_pelvis_height", 0.7)):
            reasons.append("pelvis_low")
        if tilt > float(self.ps.get("max_tilt_deg", 30.0)):
            reasons.append("tilt")
        success = bool(feet_ok and not reasons)
        com, vel = self._truth_com()
        if self.impact is not None:
            v = self.impact["v"]
            vv, vh = float(max(-v[2], 0.0)), float(np.linalg.norm(v[:2]))
            tilt0 = math.degrees(self.impact["tilt"])
        else:
            vv = vh = 0.0
            tilt0 = tilt
        a = math.radians(self.params.get("push_direction_deg", 0.0))
        return {"safe": success, "success": success,
                "impact_v_vertical": vv, "impact_v_horizontal": vh,
                "tilt_at_impact_deg": float(tilt0),
                "first_contact": self.first_contact,
                "bad_contact": bool(self.bad_contact),
                "any_bad_contact": bool(self.any_bad_contact),
                "fell": not success,
                "non_foot_contact": bool(self.non_foot_contact),
                "non_foot_body": self.non_foot_body,
                "t_fall": self.t_fall,
                "final_tilt_deg": float(tilt),
                "final_pelvis_height": ph,
                "final_com_offset_y": float(com[1]),
                "final_com_offset_x": float(com[0]),
                "final_com_speed": float(np.linalg.norm(vel)),
                "push_direction_deg": float(self.params.get("push_direction_deg", 0.0)),
                "push_lateral": float(abs(math.sin(a))),
                "failure_reasons": reasons}

    def _reward(self, terminal: bool, info: dict) -> float:
        if not terminal:
            return 0.0
        rw = self.rw
        r = float(rw.get("success_bonus", 10.0)) * float(info["success"])
        r -= float(rw.get("fall_penalty", 10.0)) * float(not info["success"])
        r -= float(rw.get("k_impulse", 0.0)) * (info["impulse_gas_pods"] + info["impulse_gas_jets"])
        return r
