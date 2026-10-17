"""Tests for suitsim.viz (offscreen rendering, mp4 export, episode plots)."""
from suitsim.viz.gl import ensure_gl

GL_BACKEND = ensure_gl()   # before mujoco is imported

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from suitsim.eval.batch import run_episode  # noqa: E402
from suitsim.eval.fake_env import FakeDropEnv  # noqa: E402
from suitsim.viz.gl import gl_available  # noqa: E402
from suitsim.viz.plots import plot_episode, trajectory_arrays  # noqa: E402

HAVE_GL = gl_available()
needs_gl = pytest.mark.skipif(not HAVE_GL, reason=f"no offscreen GL backend "
                              f"(MUJOCO_GL={GL_BACKEND}); install libegl1 or libosmesa6")
FACTORY = "suitsim.eval.fake_env:make_fake_env"
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


@needs_gl
def test_recorder_frames_and_mp4(tmp_path):
    from suitsim.viz.render import EpisodeRecorder, count_video_frames

    env = FakeDropEnv({}, log_trajectory=True)
    rec = EpisodeRecorder(env, width=160, height=128, fps=25)
    try:
        assert rec.render_every() == 4            # dt 0.01 s, 25 fps
        res = rec.record(seed=5)
        assert res["n_frames"] == len(rec.frames) > 3
        f = rec.frames[0]
        assert f.shape == (128, 160, 3) and f.dtype == np.uint8 and f.std() > 1.0
        # frames are not all identical (the box falls)
        assert np.abs(rec.frames[0].astype(int) - rec.frames[-1].astype(int)).mean() > 0.5
        mp4 = rec.save_mp4(tmp_path / "ep.mp4")
        assert mp4.stat().st_size > 1000
        assert count_video_frames(mp4) == res["n_frames"]
    finally:
        rec.close()
    # recording is a deterministic re-run of the batch episode
    ref = run_episode(FakeDropEnv({}), None, 5)
    for k in ("impact_v_vertical", "tilt_at_impact_deg", "impulse_gas_pods", "success"):
        assert res[k] == ref[k]
    assert "trajectory" in res and len(res["trajectory"]) == res["steps"]


@needs_gl
def test_plumes_and_hooks():
    import mujoco

    from suitsim.viz.render import EpisodeRecorder, find_thruster_bank, thruster_plumes

    env = FakeDropEnv({})
    env.reset(seed=1)
    for _ in range(60):          # fall until the baseline starts braking
        env.step(np.zeros(6))
        if env.thrusters.forces[0] > 0:
            break
    bank = find_thruster_bank(env)
    plumes = thruster_plumes(env.model, env.data, bank)
    assert len(plumes) == 1 and 0 < plumes[0][2] <= 1 and plumes[0][1][2] < 0  # exhaust down
    seen = {}

    def hook(scene, e):
        seen["n"] = scene.ngeom

    rec = EpisodeRecorder(env, width=64, height=64, frame_hooks=[hook], plumes=False)
    rec.render_frame()
    n_without = seen["n"]
    rec.plumes = True
    rec.render_frame()
    assert seen["n"] == n_without + 1
    rec.close()
    assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "track") == -1


@needs_gl
def test_render_episode_script(tmp_path):
    sys.path.insert(0, str(SCRIPTS))
    import render_episode
    out = tmp_path / "one.mp4"
    rc = render_episode.main(["--factory", FACTORY, "--seed", "3", "--out", str(out),
                              "--width", "160", "--height", "128", "--no-shadows"])
    assert rc == 0 and out.exists() and out.with_suffix(".png").exists()
    info = json.loads(out.with_suffix(".json").read_text())
    assert info["seed"] == 3 and info["n_frames"] > 0


@needs_gl
def test_evaluate_cli_with_videos(tmp_path):
    sys.path.insert(0, str(SCRIPTS))
    import evaluate
    out = tmp_path / "ev"
    rc = evaluate.main(["--task", "configs/task_drop.yaml", "--factory", FACTORY,
                        "--episodes", "12", "--workers", "2", "--videos", "2", "--quick",
                        "--width", "160", "--height", "128", "--out", str(out)])
    assert rc == 0
    vids = sorted(p.name for p in (out / "videos").glob("*.mp4"))
    assert len(vids) == 4 and sum(v.startswith("worst") for v in vids) == 2
    index = json.loads((out / "videos" / "index.json").read_text())
    assert all(v["deterministic"] and v["frames"] > 0 for v in index)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["n"] == 12 and len(summary["videos"]) == 4


# ------------------------------------------------------------------ plots (no GL needed)
def _traj():
    env = FakeDropEnv({}, log_trajectory=True)
    return run_episode(env, None, 2, keep_trajectory=True)


def test_plot_episode_full(tmp_path):
    res = _traj()
    a = trajectory_arrays(res["trajectory"])
    assert {"t", "com_height", "vz", "vh", "tilt", "forces_by_class.lift_pod",
            "budgets.battery_wh"} <= set(a)
    p = plot_episode(res["trajectory"], tmp_path / "ep.png", info=res,
                     cfg={"safe_landing": {"max_vertical_speed": 2.0, "max_tilt_deg": 20},
                          "suit": {"budgets": {"impulse_gas_pods": 2500.0,
                                               "impulse_gas_jets": 600.0,
                                               "battery_wh": 150.0}}})
    assert p.exists() and p.stat().st_size > 5000


def test_plot_episode_missing_data(tmp_path):
    assert plot_episode(None, tmp_path / "none.png").exists()
    assert plot_episode([], tmp_path / "empty.png", info={"success": False}).exists()
    partial = [{"t": 0.01 * i, "com_height": 2 - 0.01 * i} for i in range(50)]
    partial[10]["com_vel"] = [0, 0, -1.0]          # sparse / partially missing key
    assert set(trajectory_arrays(partial)) >= {"t", "com_height", "vz"}
    assert plot_episode(partial, tmp_path / "partial.png").exists()
    no_t = [{"tilt": 0.1}, {"tilt": 0.2}]
    assert trajectory_arrays(no_t)["t"].tolist() == [0.0, 1.0]
    assert plot_episode(no_t, tmp_path / "no_t.png").exists()
