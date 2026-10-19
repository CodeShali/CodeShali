"""First-principles hardware sizing: mission requirements -> component specs and mass.

See ``scripts/size_suit.py`` and ``docs/hardware_sizing.md``; assumptions live in
``configs/sizing.yaml``.
"""
from . import physics
from .model import (CASES, DEFAULT_CONFIG, load_sim_usage, load_sizing_config, pick,
                    requirements, run, size_architecture, verdict)

__all__ = ["physics", "CASES", "DEFAULT_CONFIG", "load_sim_usage", "load_sizing_config", "pick",
           "requirements", "run", "size_architecture", "verdict"]
