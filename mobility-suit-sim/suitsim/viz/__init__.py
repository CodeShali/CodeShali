"""Visualisation: offscreen MuJoCo rendering (render.py) and episode plots (plots.py).

Importing this package selects a working headless GL backend (see :mod:`suitsim.viz.gl`);
do it before anything imports ``mujoco``.
"""
from .gl import ensure_gl

ensure_gl()
