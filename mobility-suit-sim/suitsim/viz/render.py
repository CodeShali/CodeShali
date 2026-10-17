"""Deterministic episode re-run + offscreen MuJoCo rendering to mp4.

    rec = EpisodeRecorder(env, policy_fn=None, fps=30)
    result = rec.record(seed=123)          # re-runs the episode, keeps frames + info
    rec.save_mp4("worst_0.mp4")

Rendering uses its own :class:`mujoco.Renderer` on ``env.unwrapped.model/data`` (the env's
``render()`` is not needed). Camera: the model camera named ``track`` if present, else a
free camera that follows the pelvis (root body). Thruster plumes are drawn as translucent
capsules along the exhaust direction with length proportional to force / f_max.
"""
from __future__ import annotations

from .gl import ensure_gl

ensure_gl()  # must precede the first `import mujoco`

from pathlib import Path  # noqa: E402
from typing import Any, Callable, Sequence  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

FrameHook = Callable[[Any, Any], None]   # hook(scene: mujoco.MjvScene, env) -> None


# --------------------------------------------------------------------------------------
# Telemetry helpers
# --------------------------------------------------------------------------------------
def find_thruster_bank(env) -> Any | None:
    """Locate a ThrusterBank-like object (``.specs``, ``.forces``) on the env."""
    u = getattr(env, "unwrapped", env)
    for name in ("thrusters", "thruster_bank", "bank", "actuators"):
        b = getattr(u, name, None)
        if b is not None and hasattr(b, "specs") and hasattr(b, "forces"):
            return b
    return None


def thruster_plumes(model, data, bank) -> list[tuple[np.ndarray, np.ndarray, float, str]]:
    """[(site_pos, exhaust_dir (unit, world), force_fraction, cls)] for active thrusters."""
    if bank is None:
        return []
    forces = np.asarray(bank.forces, dtype=float).reshape(-1)
    fmax = np.asarray(getattr(bank, "f_max", [getattr(s, "f_max", 1.0) for s in bank.specs]),
                      dtype=float).reshape(-1)
    out = []
    for i, spec in enumerate(bank.specs):
        if i >= forces.size or forces[i] <= 1e-6:
            continue
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, getattr(spec, "site",
                                                                       f"thr_{spec.name}"))
        if sid < 0:
            continue
        pos = data.site_xpos[sid].copy()
        zax = data.site_xmat[sid].reshape(3, 3)[:, 2]          # site z = force direction
        frac = float(np.clip(forces[i] / max(fmax[i] if i < fmax.size else 1.0, 1e-9), 0, 1))
        out.append((pos, -zax, frac, getattr(spec, "cls", "")))
    return out


_PLUME_RGBA = {"lift_pod": (1.0, 0.45, 0.05, 0.55), "fan": (0.3, 0.7, 1.0, 0.45),
               "micro_jet": (1.0, 0.9, 0.3, 0.6)}
_PLUME_LEN = {"lift_pod": 0.8, "fan": 0.4, "micro_jet": 0.15}
_PLUME_W = {"lift_pod": 0.05, "fan": 0.04, "micro_jet": 0.012}


def add_plume_geoms(scene, plumes) -> None:
    """Append plume capsules to ``scene`` (user geoms; respects scene.maxgeom)."""
    for pos, d, frac, cls in plumes:
        if scene.ngeom >= scene.maxgeom:
            return
        length = max(1e-3, _PLUME_LEN.get(cls, 0.4) * frac)
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                            np.zeros(9), np.array(_PLUME_RGBA.get(cls, (1, .5, 0, .5)),
                                                  dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, _PLUME_W.get(cls, 0.03),
                             pos, pos + d * length)
        scene.ngeom += 1


# --------------------------------------------------------------------------------------
# Recorder
# --------------------------------------------------------------------------------------
class EpisodeRecorder:
    """Re-run an episode deterministically (``env.reset(seed)``) and render frames."""

    def __init__(self, env, policy_fn: Callable | None = None, width: int = 640,
                 height: int = 480, fps: float | None = None, camera: str | None = "track",
                 plumes: bool = True, frame_hooks: Sequence[FrameHook] = (),
                 track_body: str = "pelvis", distance: float = 4.0,
                 azimuth: float = 135.0, elevation: float = -15.0,
                 max_steps: int = 100_000, shadows: bool = True):
        self.env = env
        self.policy_fn = policy_fn
        self.width, self.height = int(width), int(height)
        cfg = getattr(env.unwrapped, "cfg", None) or {}
        self.fps = float(fps or (cfg.get("viz") or {}).get("fps", 0)
                         or env.metadata.get("render_fps", 0) or 50)
        self.camera_name = camera
        self.plumes = plumes
        self.frame_hooks = list(frame_hooks)
        self.track_body = track_body
        self.distance, self.azimuth, self.elevation = distance, azimuth, elevation
        self.max_steps = max_steps
        self.shadows = shadows
        self.frames: list[np.ndarray] = []
        self.info: dict = {}
        self._renderer = None

    # -------------------------------------------------------------- internals
    @property
    def model(self):
        return self.env.unwrapped.model

    @property
    def data(self):
        return self.env.unwrapped.data

    def env_dt(self) -> float:
        u = self.env.unwrapped
        dt = getattr(u, "dt", None) or getattr(u, "control_dt", None)
        if not dt:
            dt = ((getattr(u, "cfg", None) or {}).get("sim") or {}).get("control_dt")
        return float(dt or self.model.opt.timestep)

    def render_every(self) -> int:
        """Render one frame every N env steps so the video plays at real time."""
        return max(1, int(round(1.0 / (self.fps * self.env_dt()))))

    def _ensure_renderer(self):
        if self._renderer is not None and self._renderer_model is self.model:
            return
        self.close()
        m = self.model
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), self.width)
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), self.height)
        self._renderer = mujoco.Renderer(m, self.height, self.width)
        if not self.shadows:
            self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self._renderer_model = m
        cam_id = -1
        if self.camera_name:
            cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        self._cam_id = cam_id
        self._cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(m, self._cam)
        self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._cam.distance, self._cam.azimuth = self.distance, self.azimuth
        self._cam.elevation = self.elevation
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self.track_body)
        self._track_id = bid if bid >= 0 else min(1, m.nbody - 1)
        self._bank = find_thruster_bank(self.env)

    def render_frame(self) -> np.ndarray:
        """Render the current env state (tracking camera + plumes + hooks)."""
        self._ensure_renderer()
        r = self._renderer
        if self._cam_id >= 0:
            r.update_scene(self.data, camera=self._cam_id)
        else:
            look = self.data.xpos[self._track_id].copy()
            look[2] = max(look[2] * 0.8, 0.6)       # keep the ground in view near landing
            self._cam.lookat[:] = look
            r.update_scene(self.data, camera=self._cam)
        if self.plumes:
            add_plume_geoms(r.scene, thruster_plumes(self.model, self.data, self._bank))
        for hook in self.frame_hooks:
            hook(r.scene, self.env)
        return r.render().copy()

    # -------------------------------------------------------------- public API
    def record(self, seed: int, max_frames: int | None = None) -> dict:
        """Reset with ``seed``, run to the end, render every N steps. Returns the result
        dict from :func:`suitsim.eval.batch.run_episode` plus ``n_frames``."""
        from suitsim.eval.batch import run_episode

        self.frames = []
        every = self.render_every()
        counter = {"i": 0}

        def cb(env, obs, info):
            counter["i"] += 1
            if counter["i"] % every == 0 and (max_frames is None or len(self.frames) < max_frames):
                self.frames.append(self.render_frame())

        res = run_episode(self.env, self.policy_fn, seed, self.max_steps,
                          keep_trajectory=True, step_callback=cb,
                          reset_callback=lambda env, obs: self.frames.append(self.render_frame()))
        self.frames.append(self.render_frame())
        res["n_frames"] = len(self.frames)
        self.info = res
        return res

    def save_mp4(self, path: str | Path, frames: Sequence[np.ndarray] | None = None) -> Path:
        return save_video(frames if frames is not None else self.frames, path, self.fps)

    def close(self):
        if getattr(self, "_renderer", None) is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
        self._renderer = None
        self._renderer_model = None

    def __del__(self):
        self.close()


def save_video(frames: Sequence[np.ndarray], path: str | Path, fps: float = 50) -> Path:
    """Write frames (H,W,3 uint8) to mp4 (libx264) — or .gif if the suffix says so."""
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not len(frames):
        raise ValueError("no frames to write")
    if path.suffix.lower() == ".gif":
        imageio.mimsave(path, list(frames), duration=1.0 / fps)
        return path
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=7,
                            macro_block_size=16, pixelformat="yuv420p") as w:
        for f in frames:
            w.append_data(np.asarray(f, dtype=np.uint8))
    return path


def render_episode(env, seed: int, out_path: str | Path, policy_fn: Callable | None = None,
                   **recorder_kw) -> dict:
    """Convenience: record ``seed`` and save to ``out_path``. Returns the episode result."""
    rec = EpisodeRecorder(env, policy_fn, **recorder_kw)
    try:
        res = rec.record(seed)
        rec.save_mp4(out_path)
        res["video"] = str(out_path)
    finally:
        rec.close()
    return res


def count_video_frames(path: str | Path) -> int:
    import imageio.v2 as imageio
    n = 0
    with imageio.get_reader(path) as r:
        for _ in r:
            n += 1
    return n


__all__ = ["EpisodeRecorder", "render_episode", "save_video", "count_video_frames",
           "find_thruster_bank", "thruster_plumes", "add_plume_geoms"]
