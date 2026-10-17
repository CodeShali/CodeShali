"""Tests for suitsim.eval (metrics, batch runner, reports) using the FakeDropEnv."""
from suitsim.viz.gl import ensure_gl

ensure_gl()

import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from suitsim.eval.batch import EnvSpec, results_equal, run_episode, run_episodes  # noqa: E402
from suitsim.eval.fake_env import FakeDropEnv  # noqa: E402
from suitsim.eval.metrics import (describe, episode_score, failure_flags,  # noqa: E402
                                  primary_failure, success_by_param, summarize, wilson_ci,
                                  worst_best)
from suitsim.eval.report import (md_table, write_eval_outputs,  # noqa: E402
                                 write_markdown_report)
from suitsim.interfaces import EPISODE_INFO_KEYS  # noqa: E402

FACTORY = "suitsim.eval.fake_env:make_fake_env"
CFG = {"safe_landing": {"max_vertical_speed": 2.0, "max_horizontal_speed": 1.0,
                        "max_tilt_deg": 20.0},
       "suit": {"budgets": {"impulse_gas_pods": 2500.0, "impulse_gas_jets": 600.0,
                            "battery_wh": 150.0}},
       "task": {"name": "drop_landing", "drop_height": [1.0, 5.0]}}


def info(**kw):
    base = {"safe": True, "impact_v_vertical": 1.0, "impact_v_horizontal": 0.2,
            "tilt_at_impact_deg": 5.0, "first_contact": "foot_l", "bad_contact": False,
            "impulse_gas_pods": 100.0, "impulse_gas_jets": 10.0, "battery_wh": 1.0,
            "airbags_deployed": False, "success": True, "episode_time": 2.0}
    base.update(kw)
    if "safe" in kw and "success" not in kw:
        base["success"] = kw["safe"]
    return base


# ------------------------------------------------------------------ metrics
def test_wilson_ci_reference_values():
    lo, hi = wilson_ci(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-4)
    lo, hi = wilson_ci(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.7225, abs=1e-4)
    assert wilson_ci(0, 0) == (0.0, 1.0)
    lo, hi = wilson_ci(950, 1000)
    assert lo < 0.95 < hi and hi - lo < 0.03


def test_describe():
    d = describe([1, 2, 3, 4, float("nan"), None])
    assert d["n"] == 4 and d["mean"] == 2.5 and d["p50"] == 2.5 and d["max"] == 4
    assert describe([])["mean"] is None


def test_failure_flags_and_priority():
    assert primary_failure(info()) == "none"
    assert all(not v for v in failure_flags(info()).values())
    assert primary_failure(info(safe=False, impact_v_vertical=-3.0)) == "vertical_too_fast"
    assert primary_failure(info(safe=False, impact_v_horizontal=1.5)) == "horizontal_too_fast"
    assert primary_failure(info(safe=False, tilt_at_impact_deg=25.0)) == "tilt"
    assert primary_failure(info(safe=False, first_contact="none")) == "timeout"
    # bad contact wins over everything, but the multi-label flags keep the rest
    f = failure_flags(info(safe=False, bad_contact=True, first_contact="head",
                           impact_v_vertical=5.0, tilt_at_impact_deg=40.0))
    assert f["bad_contact"] and f["vertical_too_fast"] and f["tilt"]
    assert primary_failure(info(safe=False, bad_contact=True, impact_v_vertical=5.0)) \
        == "bad_contact"
    assert primary_failure(info(safe=True, success=False)) == "task_fail"
    assert primary_failure(info(safe=False)) == "other"
    # custom limits
    assert primary_failure(info(safe=False, impact_v_vertical=1.0),
                           {"max_vertical_speed": 0.5}) == "vertical_too_fast"


def test_summarize_counts_and_budgets():
    eps = ([info()] * 6
           + [info(safe=False, impact_v_vertical=3.0)] * 2
           + [info(safe=False, bad_contact=True, first_contact="torso")]
           + [info(safe=False, first_contact="none", impulse_gas_pods=3000.0)])
    for i, e in enumerate(eps):
        e = eps[i] = dict(e, seed=i, episode_params={"drop_height": 1.0 + i * 0.4})
    s = summarize(eps, CFG)
    assert s["n"] == 10
    assert s["success"]["k"] == 6 and s["success"]["rate"] == 0.6
    lo, hi = s["success"]["ci95"]
    assert (lo, hi) == pytest.approx(wilson_ci(6, 10))
    assert s["safe"]["k"] == 6 and s["bad_contact"]["k"] == 1
    assert s["touched_down"]["k"] == 9
    p = s["failures"]["primary"]
    assert p["vertical_too_fast"] == 2 and p["bad_contact"] == 1 and p["timeout"] == 1
    assert sum(p.values()) == s["failures"]["n_fail"] == 4
    assert s["failures"]["first_contact"]["foot_l"] == 8
    bu = s["budget_usage"]
    assert bu["impulse_gas_pods"]["within"]["k"] == 9
    assert bu["all_within"]["rate"] == pytest.approx(0.9)
    assert bu["impulse_gas_pods"]["used"]["max"] == 3000.0
    assert s["stats"]["impact_v_vertical"]["max"] == 3.0
    assert s["episode_params"]["drop_height"]["n"] == 10
    json.dumps(s)  # serialisable
    rows = success_by_param(eps, "drop_height", bins=2)
    assert sum(r["n"] for r in rows) == 10


def test_summarize_reports_missing_keys():
    s = summarize([{"safe": True, "success": True}])
    assert "impact_v_vertical" in s["missing_info_keys"]


def test_worst_best_selection():
    eps = [info(seed=0, safe=False, impact_v_vertical=6.0),
           info(seed=1, impact_v_vertical=0.3, impact_v_horizontal=0.0, tilt_at_impact_deg=1.0),
           info(seed=2, safe=False, bad_contact=True, first_contact="head"),
           info(seed=3, impact_v_vertical=1.9),
           info(seed=4, safe=False, impact_v_vertical=2.5),
           info(seed=5, impact_v_vertical=0.8)]
    worst, best = worst_best(eps, k=2)
    assert [r["seed"] for r in worst] == [2, 0]
    assert [r["seed"] for r in best] == [1, 5]
    # success always ranks above failure regardless of continuous score
    assert episode_score(info(impact_v_vertical=1.99)) > episode_score(
        info(safe=False, impact_v_vertical=0.1, first_contact="torso", bad_contact=False))
    w, b = worst_best(eps, k=3)
    assert not {r["seed"] for r in w} & {r["seed"] for r in b}
    w, b = worst_best(eps, k=10)
    assert len(w) == len(b) == 6
    # custom key: rank by impulse
    w, b = worst_best([dict(e, impulse_gas_pods=float(e["seed"])) for e in eps], k=1,
                      key=lambda r: -r["impulse_gas_pods"])
    assert w[0]["seed"] == 5 and b[0]["seed"] == 0


# ------------------------------------------------------------------ batch
def test_fake_env_contract():
    env = FakeDropEnv(CFG)
    obs, _ = env.reset(seed=3)
    assert env.observation_space.contains(obs) or obs.shape == env.observation_space.shape
    res = run_episode(env, None, 3)
    assert res["complete_info"] and set(EPISODE_INFO_KEYS) <= set(res)
    assert res["seed"] == 3 and res["steps"] > 0


def test_batch_determinism_serial_and_parallel():
    spec = EnvSpec(factory=FACTORY, cfg=CFG)
    seeds = list(range(10, 22))
    a = run_episodes(spec, None, seeds)
    b = run_episodes(spec, None, seeds)
    assert [r["seed"] for r in a] == seeds
    assert results_equal(a, b, tol=0.0)
    c = run_episodes(spec, None, seeds, n_workers=2)
    assert [r["seed"] for r in c] == seeds
    assert results_equal(a, c, tol=1e-12)
    # different seeds -> different episodes
    d = run_episodes(spec, None, [s + 100 for s in seeds])
    assert not results_equal(a, d)
    # outcomes vary (the fake env must exercise both branches of the metrics)
    assert 0 < sum(r["success"] for r in a) < len(a)


def test_batch_callable_factory_and_policy():
    calls = []

    def policy(obs):
        calls.append(1)
        return np.ones(6, dtype=np.float32)

    res = run_episodes(lambda: FakeDropEnv(CFG, controller="residual"), policy, [0, 1])
    assert len(res) == 2 and len(calls) == sum(r["steps"] for r in res)
    base = run_episodes(lambda: FakeDropEnv(CFG, controller="baseline"), None, [0, 1])
    assert not results_equal(res, base)   # residual action changed the outcome
    with pytest.raises(TypeError):
        run_episodes(lambda: FakeDropEnv(CFG), None, [0, 1, 2], n_workers=2)


def test_env_spec_from_task_config():
    spec = EnvSpec(task_cfg="configs/task_drop.yaml", factory=FACTORY,
                   overrides={"task": {"drop_height": [2.0, 2.0]}})
    cfg = spec.config()
    assert cfg["task"]["name"] == "drop_landing" and cfg["task"]["drop_height"] == [2.0, 2.0]
    r = run_episodes(spec, None, [0])[0]
    assert r["episode_params"]["drop_height"] == pytest.approx(2.0)


# ------------------------------------------------------------------ reports
def test_report_files(tmp_path):
    spec = EnvSpec(factory=FACTORY, cfg=CFG)
    res = run_episodes(spec, None, range(15))
    s = summarize(res, CFG)
    files = write_eval_outputs(res, s, tmp_path, title="fake")
    for key in ("summary_json", "results_csv", "results_jsonl", "summary_md", "hist_impact",
                "hist_tilt", "hist_budgets", "failure_modes"):
        assert files[key].exists() and files[key].stat().st_size > 0, key
    assert json.loads((tmp_path / "summary.json").read_text())["n"] == 15
    rows = list(csv.DictReader(open(tmp_path / "results.csv")))
    assert len(rows) == 15 and "failure_mode" in rows[0] and "param.drop_height" in rows[0]
    assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 15
    md = (tmp_path / "summary.md").read_text()
    assert "Wilson" in md and "Failure modes" in md and "plots/hist_impact.png" in md


def test_write_markdown_report(tmp_path):
    img = tmp_path / "plots" / "x.png"
    img.parent.mkdir()
    img.write_bytes(b"\x89PNG")
    s = summarize([info(), info(safe=False, impact_v_vertical=4.0)], CFG)
    out = write_markdown_report(
        [("Intro", "Some *text*."), ("Summary", s), ("Table", [{"a": 1, "b": 2.5}]),
         ("KV", {"x": 1}), ("Figure", img), ("Mixed", ["para", img])],
        tmp_path / "report.md", title="M7 report")
    txt = out.read_text()
    assert txt.startswith("# M7 report") and "## Intro" in txt and "![x](plots/x.png)" in txt
    assert "| a | b |" in txt and "success" in txt
    assert md_table([]) .strip() == "_(no rows)_"


def test_evaluate_cli_no_videos(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import evaluate
    out = tmp_path / "ev"
    rc = evaluate.main(["--task", "configs/task_drop.yaml", "--factory", FACTORY,
                        "--episodes", "8", "--videos", "0", "--out", str(out)])
    assert rc == 0
    for f in ("summary.json", "summary.md", "results.csv", "results.jsonl", "config.yaml",
              "meta.json", "plots/hist_impact.png"):
        assert (out / f).exists(), f
    meta = json.loads((out / "meta.json").read_text())
    assert "git_hash" in meta and meta["controller"] == "baseline"


def test_residual_policy_from_sb3_zip(tmp_path):
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    model = PPO("MlpPolicy", FakeDropEnv(CFG, controller="residual"), n_steps=64,
                batch_size=32, device="cpu", seed=0)
    model.save(tmp_path / "model.zip")
    spec = EnvSpec(factory=FACTORY, cfg=CFG, controller="residual",
                   policy_path=str(tmp_path / "model.zip"))
    a = run_episodes(spec, None, [0, 1])
    b = run_episodes(spec, None, [0, 1])
    assert results_equal(a, b) and all(r["complete_info"] for r in a)
