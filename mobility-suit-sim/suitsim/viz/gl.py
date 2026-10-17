"""Headless OpenGL backend selection for MuJoCo offscreen rendering.

MuJoCo reads ``MUJOCO_GL`` once, when ``mujoco`` is first imported, so call
:func:`ensure_gl` (or import :mod:`suitsim.viz`) BEFORE anything imports ``mujoco``.

Resolution order:
1. ``MUJOCO_GL`` already set by the user -> respected as is.
2. Otherwise each candidate in :data:`CANDIDATES` is probed in a short subprocess (an
   actual 8x8 render), the first one that works is exported via ``os.environ.setdefault``
   (child processes inherit it).  The result is cached in ``SUITSIM_GL_PROBED``.

On the dev container (Ubuntu 24.04, no GPU) ``egl`` fails (no EGL device) and ``osmesa``
works after ``apt-get install libosmesa6`` (software rasteriser, ~30 ms per 320x240 frame).
"""
from __future__ import annotations

import os
import subprocess
import sys

CANDIDATES: tuple[str, ...] = ("egl", "osmesa")

_PROBE = r"""
import mujoco
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos="0 0 2"/>'
    '<geom type="box" size=".1 .1 .1"/></worldbody></mujoco>')
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 8, 8); r.update_scene(d); img = r.render()
assert img.shape == (8, 8, 3)
print("OK")
"""


def probe_backend(name: str, timeout: float = 30.0) -> bool:
    """True if MuJoCo can render offscreen with ``MUJOCO_GL=name`` (tested in a subprocess)."""
    env = dict(os.environ, MUJOCO_GL=name)
    env.pop("PYOPENGL_PLATFORM", None)
    try:
        out = subprocess.run([sys.executable, "-c", _PROBE], env=env, timeout=timeout,
                             capture_output=True, text=True)
        return out.returncode == 0 and "OK" in out.stdout
    except Exception:
        return False


def ensure_gl() -> str | None:
    """Make sure ``MUJOCO_GL`` names a working headless backend. Returns it (or None)."""
    if os.environ.get("MUJOCO_GL"):
        return os.environ["MUJOCO_GL"]
    if "mujoco" in sys.modules:
        # Too late to choose; MuJoCo already picked its default (GLFW on Linux).
        return None
    probed = os.environ.get("SUITSIM_GL_PROBED")
    if probed is None:
        probed = next((c for c in CANDIDATES if probe_backend(c)), "")
        os.environ["SUITSIM_GL_PROBED"] = probed
    if probed:
        os.environ.setdefault("MUJOCO_GL", probed)
        return probed
    return None


def gl_available() -> bool:
    """True if offscreen rendering works in THIS process (tries a tiny render)."""
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='.1'/></worldbody>"
                                           "</mujoco>")
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        r = mujoco.Renderer(m, 8, 8)
        r.update_scene(d)
        r.render()
        r.close()
        return True
    except Exception:
        return False
