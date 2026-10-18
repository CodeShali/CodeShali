"""Pytest session setup.

MuJoCo reads ``MUJOCO_GL`` once, when ``mujoco`` is first imported, and test modules are
imported alphabetically (test_actuators... before test_viz), so the headless GL backend must
be selected here, before any test module imports mujoco; otherwise the rendering tests skip.
"""
from suitsim.viz.gl import ensure_gl

ensure_gl()
