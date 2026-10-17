"""Airbag deployment toggle (contact on/off + visibility)."""
from __future__ import annotations

import mujoco

from suitsim.interfaces import AIRBAGS, airbag_geom, load_config

DEPLOYED_ALPHA = 0.55   # cosmetic


def airbag_geom_ids(model: mujoco.MjModel) -> list[int]:
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, airbag_geom(b)) for b in AIRBAGS]
    return [i for i in ids if i >= 0]


def set_airbags(model: mujoco.MjModel, deployed: bool, cfg: dict | None = None) -> None:
    """Enable (deployed=True) or disable airbag contacts. Uses the body-geom contact bits
    from ``cfg['human']['contact']`` (default config if None)."""
    c = (cfg if cfg is not None else _default_contact())
    c = c["human"]["contact"] if "human" in c else c
    for g in airbag_geom_ids(model):
        model.geom_contype[g] = c["body_contype"] if deployed else 0
        model.geom_conaffinity[g] = c["body_conaffinity"] if deployed else 0
        model.geom_rgba[g, 3] = DEPLOYED_ALPHA if deployed else 0.0


def airbags_deployed(model: mujoco.MjModel) -> bool:
    ids = airbag_geom_ids(model)
    return bool(ids) and all(model.geom_contype[g] or model.geom_conaffinity[g] for g in ids)


_CONTACT_CACHE: dict = {}


def _default_contact() -> dict:
    if not _CONTACT_CACHE:
        _CONTACT_CACHE.update(load_config()["human"]["contact"])
    return _CONTACT_CACHE
