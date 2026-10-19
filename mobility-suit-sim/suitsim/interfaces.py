"""Shared interfaces for every workstream.

This module is the contract between the model, actuator, sensor, estimation, control,
env and eval packages. Change it only deliberately: names here are used as MJCF site /
body / geom names and as dict keys throughout the code base.

Conventions
-----------
* World frame: z up, gravity along -z.  Body frames: x forward, y left, z up (standing).
* Quaternions are MuJoCo order ``[w, x, y, z]``.
* Wrenches are ``[Fx, Fy, Fz, Tx, Ty, Tz]`` in a *world-aligned* frame whose origin is the
  (estimated) whole-body centre of mass.
* Thruster commands / forces are non-negative magnitudes along the thruster's ``dir``.
* Controllers consume ONLY :data:`SENSOR_KEYS` / :data:`ESTIMATE_KEYS` dicts — never MuJoCo
  ground truth.  Ground truth is for metrics (``info``) only.
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import yaml

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
ASSET_DIR = PROJECT_ROOT / "assets"
DEFAULT_MJCF = ASSET_DIR / "human_suit.xml"

# --------------------------------------------------------------------------------------
# Body segments (15) — also MJCF body names. IMU on each.
# --------------------------------------------------------------------------------------
SEGMENTS: tuple[str, ...] = (
    "pelvis", "torso", "head",
    "upper_arm_l", "upper_arm_r", "lower_arm_l", "lower_arm_r", "hand_l", "hand_r",
    "thigh_l", "thigh_r", "shin_l", "shin_r", "foot_l", "foot_r",
)
ROOT_BODY = "pelvis"            # free joint lives here
TORSO_BODY = "torso"            # attitude-controlled segment
FEET = ("foot_l", "foot_r")
HEAD_TORSO_BODIES = ("head", "torso", "pelvis")   # contact with ground before feet => failure

THRUSTER_CLASSES: tuple[str, ...] = ("lift_pod", "fan", "micro_jet")
AIRBAGS: tuple[str, ...] = ("collar", "chest", "hips")
BUDGET_KEYS: tuple[str, ...] = ("impulse_gas_pods", "impulse_gas_jets", "battery_wh")

# --------------------------------------------------------------------------------------
# MJCF naming helpers
# --------------------------------------------------------------------------------------
def thruster_site(name: str) -> str:
    """MJCF site name of a thruster mount (site z-axis = force direction)."""
    return f"thr_{name}"


def imu_site(segment: str) -> str:
    return f"imu_{segment}"


def imu_accel_sensor(segment: str) -> str:
    return f"acc_{segment}"


def imu_gyro_sensor(segment: str) -> str:
    return f"gyro_{segment}"


ALTIMETER_SITE = "altimeter"


def foot_sole_site(foot: str) -> str:
    return f"sole_{foot}"


def foot_touch_sensor(foot: str) -> str:
    return f"touch_{foot}"


def airbag_geom(bag: str) -> str:
    return f"airbag_{bag}"


GROUND_GEOM = "ground"

# --------------------------------------------------------------------------------------
# Dict keys
# --------------------------------------------------------------------------------------
# SensorSuite.read() -> dict with exactly these keys
SENSOR_KEYS: dict[str, str] = {
    "t": "float, time stamp of the reading [s]",
    "imu_accel": "(15,3) specific force per segment, segment frame [m/s^2]",
    "imu_gyro": "(15,3) angular velocity per segment, segment frame [rad/s]",
    "altimeter": "float, slant range along the altimeter ray [m] (max_range if no hit)",
    "foot_pressure": "(2,) normal force on soles [N], order FEET",
    "joint_pos": "(n_joints,) hinge joint angles [rad], order of model hinge joints",
    "joint_vel": "(n_joints,) hinge joint velocities [rad/s]",
    "thruster_force": "(44,) actual thruster forces [N], order of thruster specs",
    "budgets": "dict BUDGET_KEYS -> remaining amount",
}

# StateEstimator.update() -> dict with exactly these keys
ESTIMATE_KEYS: dict[str, str] = {
    "t": "float [s]",
    "com_height": "float, CoM height above ground [m]",
    "feet_height": "float, lowest sole height above ground [m] (from kinematics + height)",
    "com_vel": "(3,) CoM velocity, world frame [m/s]",
    "torso_quat": "(4,) torso orientation world<-body, wxyz",
    "torso_angvel": "(3,) torso angular velocity, body frame [rad/s]",
    "tilt": "float, angle between torso z axis and world z [rad]",
    "joint_pos": "(n_joints,) [rad]",
    "joint_vel": "(n_joints,) [rad/s]",
    "budgets": "dict BUDGET_KEYS -> remaining amount",
    "free_fall": "bool, fall detected (latched)",
    "time_since_fall": "float [s], 0 before detection",
    "foot_contact": "(2,) bool per foot",
}

# Episode info keys every env must put in `info` at episode end (ground truth metrics)
EPISODE_INFO_KEYS: dict[str, str] = {
    "safe": "bool",
    "impact_v_vertical": "float [m/s] at first foot contact",
    "impact_v_horizontal": "float [m/s]",
    "tilt_at_impact_deg": "float",
    "first_contact": "str, body name that touched ground first ('none' if never)",
    "bad_contact": "bool, head/torso/pelvis touched before the feet",
    "impulse_gas_pods": "float [N*s] used",
    "impulse_gas_jets": "float [N*s] used",
    "battery_wh": "float [Wh] used",
    "airbags_deployed": "bool",
    "success": "bool, task-level pass (== safe for T1)",
    "episode_time": "float [s]",
}


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path = CONFIG_DIR / "suit_default.yaml",
                overrides: dict | None = None) -> dict:
    """Load a YAML config, resolving a ``base:`` chain (relative to the file) and merging
    ``overrides`` on top.

    ``base`` is a single path or a list of paths.  A list is merged in order: each base is
    fully resolved (its own chain included) and deep-merged over the previous ones, then the
    file itself goes on top — ``base: [task_drop.yaml, suit_realistic.yaml]`` is the drop task
    with the realistic suit (keys the later base also defines win, so put the base whose
    values must survive last).
    """
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / path
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        bases = [base] if isinstance(base, (str, Path)) else list(base)
        merged: dict = {}
        for b in bases:
            merged = deep_merge(merged, load_config(path.parent / b))
        cfg = deep_merge(merged, cfg)
    return deep_merge(cfg, overrides or {})


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


def save_run_metadata(out_dir: str | Path, cfg: dict, extra: dict | None = None) -> Path:
    """Write config + git hash next to every experiment's outputs (rule §10)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"git_hash": git_hash(), "config_hash": config_hash(cfg), **(extra or {})}
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return out_dir


# --------------------------------------------------------------------------------------
# Thruster specs
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ThrusterSpec:
    name: str
    cls: str                 # one of THRUSTER_CLASSES
    body: str                # one of SEGMENTS
    pos: np.ndarray          # (3,) mount position, body frame [m]
    dir: np.ndarray          # (3,) unit force direction, body frame
    f_max: float             # [N]
    tau: float               # [s] first-order lag time constant
    budget: str              # one of BUDGET_KEYS ('battery' maps to battery_wh)
    alloc_lambda: float      # allocation regularisation weight for this class

    @property
    def site(self) -> str:
        return thruster_site(self.name)


def load_thruster_specs(cfg: dict) -> list[ThrusterSpec]:
    """Ordered thruster list (YAML order). This order indexes every thruster array.

    A thruster entry set to ``null`` (e.g. ``fan_l: null`` in a variant config) is removed;
    so is every thruster whose class is ``null`` / absent from ``actuator_classes`` or has
    ``count: 0``.
    """
    classes = cfg["suit"]["actuator_classes"]
    specs = []
    for name, t in cfg["suit"]["thrusters"].items():
        if t is None:
            continue
        c = classes.get(t["cls"])
        if c is None or int(c.get("count", 1)) == 0:
            continue
        d = np.asarray(t["dir"], dtype=float)
        specs.append(ThrusterSpec(
            name=name, cls=t["cls"], body=t["body"],
            pos=np.asarray(t["pos"], dtype=float), dir=d / np.linalg.norm(d),
            f_max=float(t.get("f_max", c["f_max"])), tau=float(t.get("tau", c["tau"])),
            budget="battery_wh" if c["budget"] == "battery" else c["budget"],
            alloc_lambda=float(c["alloc_lambda"]),
        ))
    return specs


G0 = 9.80665   # [m/s^2] standard gravity: propellant mass = impulse / (Isp * G0)


@dataclass(frozen=True)
class PropellantSpec:
    """Propellant of one gas budget (``suit.propellant.<budget>``)."""
    budget: str              # one of BUDGET_KEYS (gas budgets only)
    body: str                # segment carrying the tank / charges
    isp: float               # [s] specific impulse (propellant_isp)
    mass: float              # [kg] full-load propellant mass = capacity / (isp * G0)


def propellant_specs(cfg: dict) -> list[PropellantSpec]:
    """Propellant carried for each gas budget listed under ``suit.propellant`` (empty if the
    section is absent — the default suit models no propellant mass).

    ``suit.propellant.<budget> = {propellant_isp: [s], body: segment}``; the full-load mass is
    the budget capacity divided by ``propellant_isp * G0``.  The capacity of a solid-pulse
    budget is its total charge impulse (see :func:`budget_capacity`).
    """
    suit = cfg["suit"]
    out = []
    for key, p in (suit.get("propellant") or {}).items():
        if p is None:
            continue
        if key not in BUDGET_KEYS or key == "battery_wh":
            raise ValueError(f"suit.propellant: {key!r} is not a gas budget")
        isp = float(p["propellant_isp"])
        out.append(PropellantSpec(budget=key, body=str(p["body"]), isp=isp,
                                  mass=budget_capacity(cfg, key) / (isp * G0)))
    return out


def propellant_tracking(cfg: dict) -> bool:
    """``suit.propellant_mass_tracking``: deplete the propellant mass as impulse is used."""
    return bool(cfg["suit"].get("propellant_mass_tracking", False))


def solid_pulse_total_impulse(cfg: dict, budget: str) -> float | None:
    """Total charge impulse [N*s] of the solid-pulse thrusters on ``budget`` (None if the
    budget is not fed by solid-pulse classes)."""
    classes = cfg["suit"]["actuator_classes"]
    total, any_pulse, other = 0.0, False, False
    for t in (cfg["suit"]["thrusters"] or {}).values():
        if t is None:
            continue
        c = classes.get(t["cls"])
        if c is None or int(c.get("count", 1)) == 0 or c["budget"] != budget:
            continue
        sp = c.get("solid_pulses")
        if sp:
            any_pulse = True
            total += int(sp["charges_per_pod"]) * float(sp["charge_impulse"])
        else:
            other = True
    if any_pulse and other:
        raise ValueError(f"budget {budget!r} mixes solid-pulse and throttled thrusters")
    return total if any_pulse else None


def budget_capacity(cfg: dict, key: str) -> float:
    """Capacity of a budget: ``suit.budgets[key]``, except that a budget fed only by
    solid-pulse thrusters holds exactly its charges (charges_per_pod * charge_impulse * pods)."""
    sp = solid_pulse_total_impulse(cfg, key) if key != "battery_wh" else None
    return float(sp) if sp is not None else float(cfg["suit"]["budgets"][key])


def suit_mass_items(cfg: dict) -> list[tuple[str, str, float]]:
    """``(item, body, mass_kg)`` of every suit point mass: ``suit.masses`` entries split evenly
    over their bodies (``null`` entries are skipped) plus one ``propellant_<budget>`` item per
    :func:`propellant_specs` entry (full load)."""
    out = []
    for item, spec in cfg["suit"]["masses"].items():
        if spec is None:
            continue
        share = float(spec["total"]) / len(spec["bodies"])
        out += [(item, b, share) for b in spec["bodies"]]
    for p in propellant_specs(cfg):
        out.append((f"propellant_{p.budget}", p.body, p.mass))
    return out


# --------------------------------------------------------------------------------------
# Component protocols (implementations live in their own packages)
# --------------------------------------------------------------------------------------
@runtime_checkable
class ThrusterBankProtocol(Protocol):
    """suitsim.actuators.ThrusterBank(specs, cfg, dt)"""
    specs: list[ThrusterSpec]
    def reset(self, scale_fmax: np.ndarray | None = None,
              scale_tau: np.ndarray | None = None) -> None: ...
    def step(self, f_cmd: np.ndarray) -> np.ndarray:
        """Advance one physics step dt; returns actual forces (N,). Deducts budgets."""
    def remaining(self) -> dict[str, float]: ...
    def used(self) -> dict[str, float]: ...
    @property
    def forces(self) -> np.ndarray: ...
    @property
    def u_max(self) -> np.ndarray:
        """Currently available max force per thruster (0 when its budget is empty)."""


@runtime_checkable
class SensorSuiteProtocol(Protocol):
    """suitsim.sensors.SensorSuite(model, cfg, rng)"""
    def reset(self, model: Any, data: Any, noise_scale: float = 1.0) -> None: ...
    def record(self, model: Any, data: Any, thrusters: ThrusterBankProtocol) -> None:
        """Called every physics step; buffers true values for delay modelling."""
    def read(self) -> dict:
        """Delayed, noisy measurements, keys == SENSOR_KEYS."""


@runtime_checkable
class EstimatorProtocol(Protocol):
    """suitsim.estimation.StateEstimator(model, cfg)"""
    def reset(self, init_hint: dict, rng: np.random.Generator) -> None:
        """init_hint: {'torso_quat','com_vel','com_height'} truth + configured init errors."""
    def update(self, sensors: dict, dt: float) -> dict:
        """Returns dict keys == ESTIMATE_KEYS."""
    def thruster_geometry(self, est: dict) -> tuple[np.ndarray, np.ndarray]:
        """(N,3) thruster positions relative to estimated CoM and (N,3) unit force
        directions, both world-aligned, from estimated attitude + measured joint angles."""


@runtime_checkable
class AllocatorProtocol(Protocol):
    """suitsim.control.allocation.Allocator(specs, cfg)"""
    def matrix(self, r: np.ndarray, d: np.ndarray) -> np.ndarray:
        """(6,N) allocation matrix: column i = [d_i, r_i x d_i]."""
    def allocate(self, wrench: np.ndarray, r: np.ndarray, d: np.ndarray,
                 u_max: np.ndarray) -> np.ndarray:
        """min ||A u - w||^2 + sum lambda_i (u_i/u_max_i)^2, 0<=u<=u_max."""


@dataclass
class ControlOutput:
    wrench: np.ndarray                    # (6,) desired wrench about CoM, world-aligned
    thrust_cmd: np.ndarray | None = None  # (N,) if the controller allocates itself
    deploy_airbags: bool = False
    phase: str = "idle"
    debug: dict = field(default_factory=dict)


@runtime_checkable
class ControllerProtocol(Protocol):
    def reset(self) -> None: ...
    def compute(self, est: dict, sensors: dict) -> ControlOutput: ...


class EMGIntentStub:
    """Phase-4 placeholder for EMG intent sensing (out of scope this phase)."""

    def __init__(self, cfg: dict | None = None):
        self.enabled = bool((cfg or {}).get("emg", {}).get("enabled", False))

    def read(self) -> dict:
        return {"intent": "none", "confidence": 0.0}
