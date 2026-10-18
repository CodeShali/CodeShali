"""Procedural MJCF builder for the 15-segment human + mobility suit.

Every physical quantity comes from the config (``configs/suit_default.yaml``):
``human.*`` (masses, skeleton, geometry, contact, joints, posture), ``suit.*`` (masses,
thrusters, airbags), ``sensors.altimeter`` and ``sim.*``.  Only cosmetic values (colours,
marker sizes, camera placement, textures) are defined here.

Conventions: world z up; body frames x forward, y left, z up in the standing pose, origin at
the proximal joint; limbs extend along -z.  ``compiler angle="radian"``.
"""
from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from suitsim.interfaces import (
    ALTIMETER_SITE, DEFAULT_MJCF, FEET, GROUND_GEOM, ROOT_BODY, SEGMENTS, airbag_geom,
    foot_sole_site, foot_touch_sensor, imu_accel_sensor, imu_gyro_sensor, imu_site,
    load_config, load_thruster_specs, thruster_site,
)

# ---------------------------------------------------------------------------------------
# Naming (model-internal, not part of interfaces.py)
# ---------------------------------------------------------------------------------------
FREE_JOINT = "root"
STAND_KEY = "stand"
TRACK_CAMERA = "track"
RANGEFINDER_SENSOR = "rangefinder_alt"


def segment_geom(segment: str) -> str:
    """Primary collision/inertia geom of a segment."""
    return f"geom_{segment}"


def servo_actuator(joint: str) -> str:
    return f"servo_{joint}"


def suit_mass_geom(item: str, body: str) -> str:
    return f"suitmass_{item}_{body}"


# ---------------------------------------------------------------------------------------
# Cosmetics only
# ---------------------------------------------------------------------------------------
_RGBA_SKIN = "0.85 0.70 0.58 1"
_RGBA_SUIT = "0.20 0.24 0.30 1"
_RGBA_SUITMASS = "0.9 0.6 0.1 0.6"
_RGBA_AIRBAG = "1.0 0.45 0.1 0.0"        # alpha 0 = stowed; airbags.py sets the visible alpha
_RGBA_SENSOR = "0.1 0.9 0.1 0.6"
_THRUSTER_RGBA = {"lift_pod": "0.9 0.1 0.1 1", "fan": "0.1 0.4 0.95 1",
                  "micro_jet": "0.95 0.85 0.1 1"}
_THRUSTER_SIZE = {"lift_pod": "0.03 0.04", "fan": "0.035 0.02", "micro_jet": "0.01 0.015"}
_SENSOR_MARKER = "0.01"
_GROUP_BODY, _GROUP_AIRBAG, _GROUP_THRUSTER, _GROUP_SUIT, _GROUP_SENSOR = 0, 1, 2, 3, 4
_OFFSCREEN = (1280, 960)


def _f(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in np.atleast_1d(v))


def _side_key(name: str) -> str:
    return name[:-2] if name.endswith(("_l", "_r")) else name


def segment_geometry(cfg: dict, segment: str) -> dict:
    g = cfg["human"]["geometry"]
    return g[segment] if segment in g else g[_side_key(segment)]


def geom_centre(geo: dict) -> np.ndarray:
    """Centre (body frame) of a primary geom spec."""
    if "fromto" in geo:
        ft = np.asarray(geo["fromto"], float)
        return 0.5 * (ft[:3] + ft[3:])
    return np.asarray(geo.get("pos", [0, 0, 0]), float)


def _children(cfg: dict) -> dict[str, list[str]]:
    ch: dict[str, list[str]] = {}
    for body, spec in cfg["human"]["skeleton"].items():
        ch.setdefault(spec["parent"], []).append(body)
    return ch


def _joint_specs(cfg: dict) -> dict[str, dict]:
    jc = cfg["human"]["joints"]
    defaults = jc.get("defaults", {})
    return {name: {**defaults, **spec} for name, spec in jc["dofs"].items()}


def _contact_attrs(cfg: dict) -> dict[str, str]:
    c = cfg["human"]["contact"]
    return {"contype": str(c["body_contype"]), "conaffinity": str(c["body_conaffinity"]),
            "condim": str(c["condim"])}


def _quat_from_euler_deg(euler) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.deg2rad(np.asarray(euler, float)), "xyz")
    return q


# ---------------------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------------------
def _build_tree(cfg: dict) -> ET.Element:
    hum, suit = cfg["human"], cfg["suit"]
    contact = _contact_attrs(cfg)
    joints = _joint_specs(cfg)
    joints_by_body: dict[str, list[str]] = {}
    for jn, js in joints.items():
        joints_by_body.setdefault(js["body"], []).append(jn)

    # suit masses per body
    suit_by_body: dict[str, list[tuple[str, float]]] = {}
    for item, spec in suit["masses"].items():
        share = float(spec["total"]) / len(spec["bodies"])
        for b in spec["bodies"]:
            suit_by_body.setdefault(b, []).append((item, share))
    thr_by_body: dict[str, list] = {}
    for s in load_thruster_specs(cfg):
        thr_by_body.setdefault(s.body, []).append(s)
    bags_by_body: dict[str, list[str]] = {}
    for bag in ("collar", "chest", "hips"):
        if bag in suit["airbags"]:
            bags_by_body.setdefault(suit["airbags"][bag]["body"], []).append(bag)
    alt = cfg["sensors"]["altimeter"]
    sole_margin = np.asarray(hum["geometry"]["sole_site_margin"], float)
    suit_r = float(hum["geometry"]["suit_mass_radius"])
    children = _children(cfg)

    root = ET.Element("mujoco", model="human_suit")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true", inertiafromgeom="true")
    opt = ET.SubElement(root, "option", timestep=_f(cfg["sim"]["timestep"]),
                        gravity=_f([0, 0, -cfg["sim"]["gravity"]]), integrator="implicitfast")
    # runtime-resized airbags invalidate the compile-time BVH (see sim.collision_midphase)
    if not bool(cfg["sim"].get("collision_midphase", False)):
        ET.SubElement(opt, "flag", midphase="disable")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth=str(_OFFSCREEN[0]), offheight=str(_OFFSCREEN[1]))
    ET.SubElement(vis, "quality", shadowsize="2048")
    ET.SubElement(vis, "headlight", ambient="0.3 0.3 0.3", diffuse="0.5 0.5 0.5")

    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="gradient",
                  rgb1="0.45 0.65 0.9", rgb2="0.05 0.1 0.2", width="512", height="512")
    ET.SubElement(asset, "texture", name="checker", type="2d", builtin="checker",
                  rgb1="0.3 0.35 0.4", rgb2="0.2 0.25 0.3", width="512", height="512",
                  mark="edge", markrgb="0.6 0.6 0.6")
    ET.SubElement(asset, "material", name="groundmat", texture="checker",
                  texrepeat="10 10", reflectance="0.15")

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", name="sun", pos="0 0 8", dir="0 0.3 -1",
                  directional="true", castshadow="true", diffuse="0.7 0.7 0.7")
    c = hum["contact"]
    mu = float(cfg["sim"]["ground_friction"])
    ground_friction = _f([mu, 0.005, 0.0001])    # sliding (config), torsional, rolling
    ET.SubElement(world, "geom", name=GROUND_GEOM, type="plane", size="50 50 0.1",
                  material="groundmat", friction=ground_friction,
                  contype=str(c["ground_contype"]), conaffinity=str(c["ground_conaffinity"]),
                  priority=str(c["ground_priority"]), condim=str(c["condim"]))
    ET.SubElement(world, "camera", name="side", pos="0 -6 1.5", xyaxes="1 0 0 0 0.15 1")

    total_human = float(hum["mass"])

    def add_body(parent_el: ET.Element, name: str) -> None:
        sk = hum["skeleton"][name]
        b = ET.SubElement(parent_el, "body", name=name, pos=_f(sk["pos"]))
        if name == ROOT_BODY:
            ET.SubElement(b, "freejoint", name=FREE_JOINT)
            ET.SubElement(b, "camera", name=TRACK_CAMERA, mode="trackcom",
                          pos="0.5 -4.0 0.4", xyaxes="1 0.12 0 0 0.1 1")
        for jn in joints_by_body.get(name, []):
            js = joints[jn]
            ET.SubElement(b, "joint", name=jn, type="hinge", axis=_f(js["axis"]),
                          range=_f(np.deg2rad(js["range"])), damping=_f(js["damping"]),
                          armature=_f(js["armature"]), frictionloss=_f(js["frictionloss"]))
        # primary geom with the human mass of this segment
        geo = segment_geometry(cfg, name)
        attrs = {"name": segment_geom(name), "type": geo["type"], "size": _f(geo["size"]),
                 "mass": _f(total_human * float(hum["mass_frac"][name])),
                 "rgba": _RGBA_SKIN if name in ("head", "hand_l", "hand_r") else _RGBA_SUIT,
                 "group": str(_GROUP_BODY), **contact}
        if "fromto" in geo:
            attrs["fromto"] = _f(geo["fromto"])
        else:
            attrs["pos"] = _f(geo.get("pos", [0, 0, 0]))
        ET.SubElement(b, "geom", **attrs)
        centre = geom_centre(geo)
        # suit masses, co-located with the primary geom centre (keeps the segment CoM)
        for item, m in suit_by_body.get(name, []):
            ET.SubElement(b, "geom", name=suit_mass_geom(item, name), type="sphere",
                          size=_f(suit_r), pos=_f(centre), mass=_f(m), contype="0",
                          conaffinity="0", group=str(_GROUP_SUIT), rgba=_RGBA_SUITMASS)
        # airbags (massless). Compiled WITH the body contact bits (MuJoCo fixes part of the
        # collision filtering at compile time) and in the stowed geometry: shrunk by
        # ``stowed_scale`` at the segment geom centre, deep inside the body.  Deployment is
        # by geometry at runtime (suitsim.model.airbags).  Priority above the ground's so
        # the airbag's soft solref/solimp govern its ground contacts (MuJoCo takes contact
        # parameters from the higher-priority geom); its friction mirrors the ground's
        # (kept in sync by ModelRandomizer).
        stowed_scale = float(suit["airbags"]["stowed_scale"])
        for bag in bags_by_body.get(name, []):
            a = suit["airbags"][bag]
            ab = {"name": airbag_geom(bag), "type": a["type"],
                  "size": _f(stowed_scale * np.asarray(a["size"], float)),
                  "pos": _f(centre), "mass": "0", "contype": contact["contype"],
                  "conaffinity": contact["conaffinity"],
                  "condim": str(c["condim"]), "priority": str(int(c["ground_priority"]) + 1),
                  "friction": ground_friction,
                  "solref": _f(suit["airbags"]["solref"]),
                  "solimp": _f(suit["airbags"]["solimp"]), "rgba": _RGBA_AIRBAG,
                  "group": str(_GROUP_AIRBAG)}
            if "euler" in a:
                ab["quat"] = _f(_quat_from_euler_deg(a["euler"]))
            ET.SubElement(b, "geom", **ab)
        # IMU at the segment's geometric centre
        ET.SubElement(b, "site", name=imu_site(name), type="box", pos=_f(centre),
                      size=_f([float(_SENSOR_MARKER)] * 3), group=str(_GROUP_SENSOR),
                      rgba=_RGBA_SENSOR)
        # foot sole: touch site box covering the sole face
        if name in FEET:
            pos = np.asarray(geo["pos"], float)
            half = np.asarray(geo["size"], float)
            sole_z = pos[2] - half[2]
            ET.SubElement(b, "site", name=foot_sole_site(name), type="box",
                          pos=_f([pos[0], pos[1], sole_z]),
                          size=_f([half[0] + sole_margin[0], half[1] + sole_margin[1],
                                   sole_margin[2]]),
                          group=str(_GROUP_SENSOR), rgba=_RGBA_SENSOR)
        if name == alt["body"]:
            ET.SubElement(b, "site", name=ALTIMETER_SITE, type="sphere", pos=_f(alt["pos"]),
                          zaxis=_f(alt["dir"]), size=_SENSOR_MARKER,
                          group=str(_GROUP_SENSOR), rgba=_RGBA_SENSOR)
        for s in thr_by_body.get(name, []):
            ET.SubElement(b, "site", name=thruster_site(s.name), type="capsule",
                          pos=_f(s.pos), zaxis=_f(s.dir), size=_THRUSTER_SIZE[s.cls],
                          rgba=_THRUSTER_RGBA[s.cls], group=str(_GROUP_THRUSTER))
        for ch in children.get(name, []):
            add_body(b, ch)

    for top in children["world"]:
        add_body(world, top)

    act = ET.SubElement(root, "actuator")
    for jn, js in joints.items():
        rng = np.deg2rad(js["range"])
        ET.SubElement(act, "position", name=servo_actuator(jn), joint=jn, kp=_f(js["kp"]),
                      kv=_f(js["kv"]), ctrlrange=_f(rng),
                      forcerange=_f([-js["forcerange"], js["forcerange"]]))

    sens = ET.SubElement(root, "sensor")
    for seg in SEGMENTS:
        ET.SubElement(sens, "accelerometer", name=imu_accel_sensor(seg), site=imu_site(seg))
        ET.SubElement(sens, "gyro", name=imu_gyro_sensor(seg), site=imu_site(seg))
    for foot in FEET:
        ET.SubElement(sens, "touch", name=foot_touch_sensor(foot), site=foot_sole_site(foot))
    ET.SubElement(sens, "rangefinder", name=RANGEFINDER_SENSOR, site=ALTIMETER_SITE)
    return root


def build_mjcf(cfg: dict | None = None) -> str:
    """MJCF XML string for the human + suit model described by ``cfg``.

    Includes a ``stand`` keyframe: nominal posture, soles on the ground, servos at posture.
    """
    cfg = cfg if cfg is not None else load_config()
    root = _build_tree(cfg)
    # compile once to compute the standing keyframe
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    from suitsim.model.pose import set_initial_pose   # local import: avoids a cycle
    set_initial_pose(model, data, 0.0, np.array([1.0, 0, 0, 0]), np.zeros(3), cfg=cfg)
    kf = ET.SubElement(root, "keyframe")
    ET.SubElement(kf, "key", name=STAND_KEY, qpos=_f(data.qpos), ctrl=_f(data.ctrl))
    ET.indent(root, space="  ")
    header = ("<!-- GENERATED by suitsim.model.builder.write_default_asset() from "
              "configs/suit_default.yaml. Do not edit by hand. -->\n")
    return header + ET.tostring(root, encoding="unicode") + "\n"


def build_model(cfg: dict | None = None) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(build_mjcf(cfg))


def write_default_asset(path: str | Path = DEFAULT_MJCF, cfg: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_mjcf(cfg))
    return path


# ---------------------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------------------
def model_info(model: mujoco.MjModel, cfg: dict | None = None) -> dict:
    """Index bookkeeping used by sensors / estimation / control / envs.

    Keys: joint_names (hinges, qpos order), qpos_adr, dof_adr (np int arrays, same order),
    servo_ids (actuator ids, same order), servo_names, nominal_posture (rad, same order),
    free_qpos_adr, free_dof_adr, body_ids {segment: id}, root_body_id,
    thruster_site_ids (spec order, if cfg given or looked up by name from default config),
    thruster_names, imu_site_ids (SEGMENTS order), sole_site_ids (FEET order),
    altimeter_site_id, ground_geom_id, total_mass, human_mass, suit_mass.
    """
    cfg = cfg if cfg is not None else load_config()
    hinge = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    hinge.sort(key=lambda j: model.jnt_qposadr[j])
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in hinge]
    servo_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                            servo_actuator(n)) for n in names], int)
    posture = cfg["human"].get("posture", {}) or {}
    free = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, FREE_JOINT)
    specs = load_thruster_specs(cfg)

    def sid(n):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)

    body_ids = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, s) for s in SEGMENTS}
    suit_mass = float(sum(float(v["total"]) for v in cfg["suit"]["masses"].values()))
    total = float(sum(model.body_mass))
    return {
        "joint_names": names,
        "qpos_adr": np.array([model.jnt_qposadr[j] for j in hinge], int),
        "dof_adr": np.array([model.jnt_dofadr[j] for j in hinge], int),
        "servo_ids": servo_ids,
        "servo_names": [servo_actuator(n) for n in names],
        "nominal_posture": np.deg2rad([float(posture.get(n, 0.0)) for n in names]),
        "free_qpos_adr": int(model.jnt_qposadr[free]),
        "free_dof_adr": int(model.jnt_dofadr[free]),
        "body_ids": body_ids,
        "root_body_id": body_ids[ROOT_BODY],
        "thruster_names": [s.name for s in specs],
        "thruster_site_ids": np.array([sid(s.site) for s in specs], int),
        "imu_site_ids": np.array([sid(imu_site(s)) for s in SEGMENTS], int),
        "sole_site_ids": np.array([sid(foot_sole_site(f)) for f in FEET], int),
        "altimeter_site_id": sid(ALTIMETER_SITE),
        "ground_geom_id": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, GROUND_GEOM),
        "total_mass": total,
        "suit_mass": suit_mass,
        "human_mass": total - suit_mass,
    }
