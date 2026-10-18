"""T2 — boosted jump: stand on the ground, jump onto a raised platform (milestone M6).

The human starts standing (nominal posture, feet on the ground at the origin, facing +x).
A static box platform (``task.platform_height`` high, ``task.platform_size`` x/y extent)
stands with its centre ``task.platform_distance`` ahead along +x.  The suit must lift the
wearer onto the platform top and land safely.

Platform geom
-------------
Injected into the generated MJCF (``build_mjcf(cfg)``) as a world-body ``<geom
name="platform" type="box">`` with the ground's contact classes (contype/conaffinity from
``human.contact.ground_*``) so it collides with every body geom but not with the ground.
Its x position is moved per episode (``model.geom_pos``) when the distance is randomised.

Success (``info['success']``), evaluated on ground truth
-------------------------------------------------------
* the first contact after lift-off is a FOOT on the platform TOP (contact point within
  ``success.top_tol`` of the top face; a foot hitting the platform side is not a landing);
* at that first platform-foot contact the whole-body CoM velocity satisfies
  ``safe_landing`` (vertical < 2 m/s, horizontal < 1 m/s) and torso tilt < 20 deg
  (this is ``info['safe']``);
* no head/torso/pelvis contact with any world geom after lift-off (episode terminates);
* no contact with the ground after lift-off (fell off / fell short; episode terminates);
* the wearer stays on the platform for ``task.settle_time`` after touchdown: at the end
  at least one foot touches the platform, the whole-body CoM is above the platform
  footprint and the true torso tilt is below ``success.max_tilt_end_deg``.

``safe`` alone (feet-first, within the impact limits, whatever surface) vs ``success``
lets the evaluation separate "landed safely but missed the platform" (``task_fail``).
The episode also ends when the wearer never leaves the ground or never lands
(``episode_time`` cap; ``first_contact == 'none'``).
"""
from __future__ import annotations

import math
from contextlib import contextmanager

import mujoco
import numpy as np

import suitsim.envs.base as _base
from suitsim.envs.base import SuitEnvBase
from suitsim.interfaces import FEET, HEAD_TORSO_BODIES
from suitsim.model import build_mjcf, set_initial_pose

PLATFORM_GEOM = "platform"


def _yaw_quat(yaw_rad: float) -> np.ndarray:
    return np.array([math.cos(yaw_rad / 2), 0.0, 0.0, math.sin(yaw_rad / 2)])


def platform_geom_xml(cfg: dict, distance: float | None = None) -> str:
    task = cfg["task"]
    h = float(task.get("platform_height", 2.5))
    sx, sy = (float(v) for v in task.get("platform_size", [1.5, 1.5]))
    d = float(task.get("platform_distance", 1.5) if distance is None else distance)
    c = cfg["human"]["contact"]
    mu = float(cfg["sim"].get("ground_friction", 1.0))
    return (f'<geom name="{PLATFORM_GEOM}" type="box" size="{sx / 2:.6g} {sy / 2:.6g} '
            f'{h / 2:.6g}" pos="{d:.6g} 0 {h / 2:.6g}" friction="{mu:.6g} 0.005 0.0001" '
            f'contype="{c["ground_contype"]}" conaffinity="{c["ground_conaffinity"]}" '
            f'priority="{c["ground_priority"]}" condim="{c["condim"]}" '
            f'rgba="0.55 0.5 0.45 1"/>')


def build_platform_model(cfg: dict) -> mujoco.MjModel:
    """Suit model + static platform box (world body)."""
    xml = build_mjcf(cfg)
    i = xml.rfind("</worldbody>")
    if i < 0:
        raise RuntimeError("generated MJCF has no </worldbody>")
    xml = xml[:i] + "  " + platform_geom_xml(cfg) + "\n  " + xml[i:]
    return mujoco.MjModel.from_xml_string(xml)


@contextmanager
def _patched_build_model(fn):
    """SuitEnvBase.__init__ calls ``build_model(cfg)``; swap in the platform builder for the
    duration of the base constructor (keeps base.py untouched)."""
    orig = _base.build_model
    _base.build_model = fn
    try:
        yield
    finally:
        _base.build_model = orig


class BoostedJumpEnv(SuitEnvBase):
    """T2 env.  ``controller='baseline'`` uses :class:`suitsim.control.jump.BaselineJumpController`
    (the controller's ``debug['joint_targets']`` {joint: rad} are applied to the posture
    servos after every control step)."""

    def __init__(self, cfg: dict, controller: str = "baseline", render_mode: str | None = None,
                 log_trajectory: bool = False, policy_dt: float | None = None, **kw):
        with _patched_build_model(build_platform_model):
            super().__init__(cfg, controller=controller, render_mode=render_mode,
                             log_trajectory=log_trajectory, policy_dt=policy_dt, **kw)
        m = self.model
        self.platform_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, PLATFORM_GEOM)
        if self.platform_id < 0:
            raise RuntimeError("platform geom missing")
        task = cfg["task"]
        self.H = float(task.get("platform_height", 2.5))
        self.size = np.asarray(task.get("platform_size", [1.5, 1.5]), float)
        self.settle_time = float(task.get("settle_time", 1.0))
        sl = cfg["safe_landing"]
        self.lim_v = float(sl["max_vertical_speed"])
        self.lim_h = float(sl["max_horizontal_speed"])
        self.lim_tilt = float(sl["max_tilt_deg"])
        sc = task.get("success", {}) or {}
        self.top_tol = float(sc.get("top_tol", 0.03))
        self.max_tilt_end = float(sc.get("max_tilt_end_deg", 30.0))
        self.liftoff_height = float(sc.get("liftoff_height", 0.02))
        self.rw = dict(cfg.get("reward", {}) or {})
        self.nom_platform_friction = m.geom_friction[self.platform_id].copy()
        if controller != "none":
            from suitsim.control.jump import BaselineJumpController
            self.controller = BaselineJumpController(
                cfg, self.specs, geometry_fn=self.estimator.thruster_geometry)
        self._servo = {n: int(a) for n, a in zip(self.info_m["joint_names"],
                                                 self.info_m["servo_ids"]) if a >= 0}

    @property
    def dt(self) -> float:
        """Env step [s] (gym convention; the video recorder uses it for real-time playback)."""
        return self.policy_dt

    # ------------------------------------------------------------------ episode
    def _sample_episode(self, rng: np.random.Generator) -> dict:
        task = self.cfg["task"]
        fixed = task.get("fixed") or {}
        rc = self.cfg.get("randomization", {}) or {}
        on = bool(rc.get("enabled", True))
        d0 = float(task.get("platform_distance", 1.5))
        dd = float(task.get("platform_distance_range", 0.0))
        ly = float(task.get("platform_lateral_range", 0.0))
        ymax = float(task.get("initial_yaw_deg", 0.0))
        smax = float(task.get("initial_sway_deg", 0.0))
        # draw unconditionally (random stream independent of `fixed` / enabled)
        d = d0 + rng.uniform(-dd, dd)
        y = rng.uniform(-ly, ly)
        yaw = rng.uniform(-ymax, ymax)
        sway = rng.uniform(-smax, smax, 2)
        dr = self._apply_dr_common(rng)
        if not on:
            d, y, yaw, sway = d0, 0.0, 0.0, np.zeros(2)
        d = float(fixed.get("platform_distance", d))
        y = float(fixed.get("platform_lateral", y))
        yaw = float(fixed.get("yaw_deg", yaw))
        if "sway_deg" in fixed:
            sway = np.broadcast_to(np.asarray(fixed["sway_deg"], float), (2,)).copy()

        m = self.model
        m.geom_pos[self.platform_id] = [d, y, self.H / 2]
        m.geom_friction[self.platform_id] = self.nom_platform_friction
        m.geom_friction[self.platform_id, 0] = dr["friction"]
        self.platform_xy = np.array([d, y])

        # standing start: small sway (tilt about x / y) + heading, soles on the ground
        q = _yaw_quat(math.radians(yaw))
        for k, ang in enumerate(sway):
            axis = np.zeros(3)
            axis[k] = 1.0
            qs = np.concatenate([[math.cos(math.radians(ang) / 2)],
                                 math.sin(math.radians(ang) / 2) * axis])
            out = np.empty(4)
            mujoco.mju_mulQuat(out, qs, q)
            q = out
        set_initial_pose(m, self.data, 0.0, q, np.zeros(3), cfg=self.cfg)
        return {"platform_distance": d, "platform_lateral": y, "yaw_deg": yaw,
                "sway_x_deg": float(sway[0]), "sway_y_deg": float(sway[1]), **dr}

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.t_liftoff = None
        self.ground_after_liftoff = False
        self.ground_body = None
        self.plat_first = None           # dict for the first platform contact
        self.plat_foot_first = None
        self.first_any = "none"          # first body touching anything after lift-off
        self.side_hit = False
        self.bad_after = False
        self._ground_now = True
        self._plat_feet_now = False
        self._used_prev = 0.0
        if hasattr(self.controller, "set_mission"):
            self.controller.set_mission(self.platform_xy)
        return obs, info

    # ------------------------------------------------------------------ physics hooks
    def _control_step(self, action):
        super()._control_step(action)
        jt = (self.ctrl_debug or {}).get("joint_targets")
        if jt:
            for name, q in jt.items():
                a = self._servo.get(name)
                if a is not None:
                    self.data.ctrl[a] = q

    def _physics_step(self) -> None:
        self._ground_now = False
        self._plat_feet_now = False
        super()._physics_step()
        if self.t_liftoff is None and not self._ground_now:
            soles = self.data.site_xpos[self.sole_sites][:, 2]
            if soles.min() > self.liftoff_height:
                self.t_liftoff = float(self.data.time)

    def _contacts(self) -> None:
        d, m = self.data, self.model
        t = float(d.time) - self.dt_phys
        ground_bodies, plat = set(), []
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in self.env_geoms and g2 not in self.env_geoms:
                env_g, b = g1, int(m.geom_bodyid[g2])
            elif g2 in self.env_geoms and g1 not in self.env_geoms:
                env_g, b = g2, int(m.geom_bodyid[g1])
            else:
                continue
            if env_g == self.platform_id:
                plat.append((b, float(c.pos[2])))
            else:
                ground_bodies.add(b)
        if ground_bodies:
            self._ground_now = True
        airborne = self.t_liftoff is not None
        if airborne and ground_bodies:
            if not self.ground_after_liftoff:
                self.ground_body = self.body_names[min(ground_bodies)]
                if self.first_any == "none":
                    self.first_any = self.body_names[min(ground_bodies)]
            self.ground_after_liftoff = True
        if airborne and (ground_bodies & self.bad_ids):
            self.bad_after = True
        if not plat:
            return
        bodies = {b for b, _ in plat}
        feet = bodies & self.feet_ids
        bad = bodies & self.bad_ids
        top = self.H - self.top_tol
        feet_top = {b for b, z in plat if b in self.feet_ids and z >= top}
        if feet_top:
            self._plat_feet_now = True
        if bad:
            self.bad_after = True
        if self.plat_first is None:
            b = min(feet) if feet else (min(bad) if bad else min(bodies))
            self.plat_first = {"body": self.body_names[b], "t": t,
                               "top": any(z >= top for bb, z in plat if bb == b)}
            if self.first_any == "none":
                self.first_any = self.body_names[b]
            if feet and not feet_top:
                self.side_hit = True
        if feet_top and self.plat_foot_first is None:
            mujoco.mj_subtreeVel(m, d)
            vel = d.subtree_linvel[self.root_id].copy()
            self.plat_foot_first = {"v": vel, "tilt": self._truth_tilt(), "t": t,
                                    "body": self.body_names[min(feet_top)],
                                    "com": d.subtree_com[self.root_id].copy()}

    # ------------------------------------------------------------------ termination / info
    def _done(self) -> bool:
        if self.bad_after or self.ground_after_liftoff:
            return True
        pf = self.plat_foot_first
        return pf is not None and self.data.time - pf["t"] >= self.settle_time - 1e-9

    def _on_platform_now(self) -> bool:
        com, _ = self._truth_com()
        half = self.size / 2
        inside = bool(np.all(np.abs(com[:2] - self.platform_xy) <= half))
        return bool(self._plat_feet_now and inside and com[2] > self.H
                    and math.degrees(self._truth_tilt()) < self.max_tilt_end)

    def _task_info(self) -> dict:
        pf = self.plat_foot_first
        first_any = self.first_any
        if pf is not None and first_any in FEET and self.plat_first["body"] in FEET:
            v = pf["v"]
            vv, vh = float(max(-v[2], 0.0)), float(np.linalg.norm(v[:2]))
            tilt = math.degrees(pf["tilt"])
            t_imp = pf["t"]
        elif self.t_liftoff is not None and first_any != "none":
            # landed somewhere else first (ground, platform side, ...): report that impact
            mujoco.mj_subtreeVel(self.model, self.data)
            v = self.data.subtree_linvel[self.root_id]
            vv, vh = float(max(-v[2], 0.0)), float(np.linalg.norm(v[:2]))
            tilt = math.degrees(self._truth_tilt())
            t_imp = None
        else:
            vv = vh = float("nan")
            tilt = math.degrees(self._truth_tilt())
            t_imp = None
        feet_first = first_any in FEET
        bad_contact = bool(self.bad_after and not (pf is not None and feet_first))
        safe = bool(feet_first and not bad_contact and vv < self.lim_v and vh < self.lim_h
                    and tilt < self.lim_tilt)
        on_top_first = bool(pf is not None and self.plat_first is not None
                            and self.plat_first["body"] in FEET and self.plat_first["top"]
                            and first_any == self.plat_first["body"])
        stayed = bool(pf is not None and not self.ground_after_liftoff and not self.bad_after
                      and self.data.time - pf["t"] >= self.settle_time - 1e-6
                      and self._on_platform_now())
        success = bool(safe and on_top_first and stayed)
        com, _ = self._truth_com()
        reasons = []
        if self.t_liftoff is None:
            reasons.append("no_liftoff")
        elif first_any == "none":
            reasons.append("no_contact")
        elif not feet_first:
            reasons.append("bad_contact" if bad_contact else "non_foot_first")
        if feet_first and not on_top_first:
            reasons.append("platform_side" if self.side_hit else "missed_platform")
        if self.ground_after_liftoff:
            reasons.append("ground_contact")
        if self.bad_after and not bad_contact:
            reasons.append("fell_after_landing")
        if np.isfinite(vv) and not vv < self.lim_v:
            reasons.append("v_vertical")
        if np.isfinite(vh) and not vh < self.lim_h:
            reasons.append("v_horizontal")
        if first_any != "none" and not tilt < self.lim_tilt:
            reasons.append("tilt")
        if safe and on_top_first and not stayed:
            reasons.append("did_not_stay")
        return {"safe": safe, "success": success,
                "impact_v_vertical": vv, "impact_v_horizontal": vh,
                "impact_speed": float(math.hypot(vv, vh)) if np.isfinite(vv) else float("nan"),
                "tilt_at_impact_deg": float(tilt),
                "first_contact": first_any,
                "bad_contact": bad_contact,
                "any_bad_contact": bool(self.bad_after),
                "t_liftoff": self.t_liftoff,
                "t_first_contact": t_imp, "contact_time": t_imp,
                "landed_on_platform": on_top_first,
                "stayed_on_platform": stayed,
                "ground_after_liftoff": bool(self.ground_after_liftoff),
                "platform_side_hit": bool(self.side_hit),
                "landing_com_x_rel": (float(pf["com"][0] - self.platform_xy[0])
                                      if pf is not None else float("nan")),
                "landing_com_y_rel": (float(pf["com"][1] - self.platform_xy[1])
                                      if pf is not None else float("nan")),
                "final_com_x_rel": float(com[0] - self.platform_xy[0]),
                "final_com_y_rel": float(com[1] - self.platform_xy[1]),
                "failure_reasons": reasons}

    def _reward(self, terminal: bool, info: dict) -> float:
        rw = self.rw
        r = 0.0
        k_shape = float(rw.get("shaping_impulse", 0.0))
        if k_shape:
            used = self.thrusters.used()
            tot = used["impulse_gas_pods"] + used["impulse_gas_jets"]
            r -= k_shape * (tot - self._used_prev)
            self._used_prev = tot
        if not terminal:
            return r
        vv, vh = info["impact_v_vertical"], info["impact_v_horizontal"]
        speed2 = (vv * vv + vh * vh) if np.isfinite(vv) else 0.0
        r += float(rw.get("safe_bonus", 10.0)) * float(info["success"])
        r -= float(rw.get("k1_impact_speed2", 0.0)) * speed2
        r -= float(rw.get("k2_tilt", 0.0)) * math.radians(info["tilt_at_impact_deg"])
        r -= float(rw.get("k3_impulse", 0.0)) * (info["impulse_gas_pods"] + info["impulse_gas_jets"])
        if info["bad_contact"] or info["first_contact"] == "none":
            r -= float(rw.get("bad_contact_penalty", 50.0))
        return r
