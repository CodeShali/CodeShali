"""scripts/train.py smoke test: --quick with a tiny step budget, SubprocVecEnv (spawn)."""
from __future__ import annotations

import csv
import importlib.util
import json

import numpy as np
import pytest

from suitsim.interfaces import PROJECT_ROOT


def _train_module():
    spec = importlib.util.spec_from_file_location("train_script", PROJECT_ROOT / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_quick_config_resolution():
    train = _train_module()
    args = train.build_parser().parse_args(["--quick", "--seed", "3", "--set",
                                            "reward.safe_bonus=7"])
    tcfg, task = train.build_configs(args)
    assert tcfg["n_envs"] == 4 and tcfg["total_timesteps"] == 200_000 and tcfg["seed"] == 3
    assert task["reward"]["safe_bonus"] == 7
    assert task["reward"]["k3_impulse"] == 0.0 and task["reward"]["shaping_impulse"] > 0
    assert task["task"]["name"] == "drop_landing"


def test_train_quick_smoke(tmp_path):
    pytest.importorskip("stable_baselines3")
    train = _train_module()
    out = tmp_path / "run"
    rc = train.main(["--quick", "--total-timesteps", "256", "--n-envs", "2", "--seed", "1",
                     "--eval-freq", "256", "--eval-episodes", "2",
                     "--train-set", "ppo.n_steps=128", "--train-set", "ppo.batch_size=128",
                     "--train-set", "ppo.n_epochs=1", "--train-set", "eval_workers=1",
                     "--train-set", "eval_baseline=false",
                     "--set", "task.drop_height=[1.0,1.5]", "--out", str(out)])
    assert rc == 0
    for f in ("model.zip", "vecnormalize.pkl", "config.yaml", "meta.json", "timing.json",
              "eval.csv", "progress.csv", "best/model.zip", "best/vecnormalize.pkl"):
        assert (out / f).exists(), f
    meta = json.loads((out / "meta.json").read_text())
    assert "git_hash" in meta and meta["status"] == "done"
    rows = list(csv.DictReader(open(out / "eval.csv")))
    assert len(rows) == 1 and int(rows[0]["timesteps"]) == 256
    from suitsim.eval.batch import load_sb3_policy
    pol = load_sb3_policy(out / "model.zip")
    a = pol(np.zeros(47, np.float32))
    assert a.shape == (6,)
