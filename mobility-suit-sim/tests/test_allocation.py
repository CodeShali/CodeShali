"""Thrust allocation tests (spec 5.1, milestone M3).

M3 metric (block-wise, stricter than a combined norm): for each feasible target wrench
w = [F; T] and the allocated wrench A u,
    ||F_alloc - F|| <= 2% * ||F||    and    ||T_alloc - T|| <= 2% * ||T||,
with the fall-back denominator ||W w|| (W = diag(1,1,1,1/L,1/L,1/L)) for a block whose norm is
below 5% of ||W w|| (so a near-zero block is not held to a relative-to-zero standard).
"""
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from suitsim.control import Allocator
from suitsim.control.allocation import axis_authority, nominal_geometry, skeleton_origins
from suitsim.interfaces import ThrusterSpec, load_config, load_thruster_specs

TOL = 0.02


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def specs(cfg):
    return load_thruster_specs(cfg)


@pytest.fixture(scope="module")
def alloc(specs, cfg):
    return Allocator(specs, cfg)


def block_errors(al, A, u, w):
    wa = A @ u
    ww = np.linalg.norm(al.W * w)
    errs = []
    for sl in (slice(0, 3), slice(3, 6)):
        num = np.linalg.norm(al.W[sl] * (wa[sl] - w[sl]))
        blk = np.linalg.norm(al.W[sl] * w[sl])
        errs.append(num / (blk if blk >= 0.05 * ww else ww))
    return errs


def cls_mask(specs, cls):
    return np.array([s.cls == cls for s in specs])


# ----------------------------------------------------------------------------- (1)
def test_matrix_columns_handmade():
    mk = lambda n, p, d: ThrusterSpec(n, "micro_jet", "torso", np.array(p, float),
                                      np.array(d, float), 40.0, 0.005, "impulse_gas_jets", 1e-4)
    sp = [mk("a", [0, 0, 0], [0, 0, 1]), mk("b", [1, 0, 0], [0, 0, 1]),
          mk("c", [0, 2, 0], [1, 0, 0])]
    al = Allocator(sp, {})
    r = np.array([[0, 0, 0], [1.0, 0, 0], [0, 2.0, 0]])
    d = np.array([[0, 0, 1.0], [0, 0, 1.0], [1.0, 0, 0]])
    A = al.matrix(r, d)
    np.testing.assert_allclose(A[:, 0], [0, 0, 1, 0, 0, 0])
    np.testing.assert_allclose(A[:, 1], [0, 0, 1, 0, -1, 0])     # x × z = -y
    np.testing.assert_allclose(A[:, 2], [1, 0, 0, 0, 0, -2])     # 2y × x = -2z
    # and the allocator reproduces a reachable wrench exactly
    u = al.allocate(A @ np.array([10.0, 5.0, 3.0]), r, d, np.full(3, 40.0))
    np.testing.assert_allclose(u, [10.0, 5.0, 3.0], atol=1e-3)


def test_skeleton_origins(cfg):
    org = skeleton_origins(cfg)
    np.testing.assert_allclose(org["pelvis"], [0, 0, 1.0])
    np.testing.assert_allclose(org["foot_l"], [0, 0.09, 1.0 - 0.05 - 0.44 - 0.43])
    np.testing.assert_allclose(org["hand_r"], [0, -0.19, 1.0 + 0.1 + 0.42 - 0.3 - 0.26])


# ----------------------------------------------------------------------------- (2) M3
def test_m3_random_feasible_wrenches(alloc, specs, cfg):
    rng = np.random.default_rng(1234)
    worst_f = worst_t = 0.0
    n = len(specs)
    for k in range(200):
        R = Rotation.random(random_state=rng.integers(1 << 31)).as_matrix()
        r, d = nominal_geometry(specs, cfg, rotation=R)
        A = alloc.matrix(r, d)
        density = rng.choice([0.05, 0.15, 0.3, 0.6, 1.0])
        mask = rng.random(n) < density
        if k % 4 == 0:                      # large vertical: pods near full + some jets
            mask |= cls_mask(specs, "lift_pod")
        u_true = rng.uniform(0, 1, n) * alloc.scale * mask
        if not u_true.any():
            u_true[rng.integers(n)] = 10.0
        w = A @ u_true
        u = alloc.allocate(w, r, d, alloc.scale)
        ef, et = block_errors(alloc, A, u, w)
        worst_f, worst_t = max(worst_f, ef), max(worst_t, et)
        assert ef <= TOL and et <= TOL, (k, ef, et, w)
        assert alloc.last_residual <= TOL
    print(f"\nM3 worst force err {worst_f:.2e}, worst torque err {worst_t:.2e}")


# ----------------------------------------------------------------------------- (3) classes
def test_vertical_force_prefers_pods(alloc, specs, cfg):
    r, d = nominal_geometry(specs, cfg)
    A = alloc.matrix(r, d)
    w = np.array([0, 0, 1500.0, 0, 0, 0])
    u = alloc.allocate(w, r, d, alloc.scale)
    assert alloc.last_residual < 1e-3
    pods = u[cls_mask(specs, "lift_pod")].sum()
    jets = (u * A[2])[cls_mask(specs, "micro_jet")].sum()
    print(f"\n+Fz 1500 N: pods {pods:.0f} N, up-jets {jets:.0f} N (default lambdas)")
    assert pods >= 0.7 * 1500 and pods > 2.5 * jets
    # with a heavier micro-jet weight (reserve jets for torque) the pods take ~everything
    al2 = Allocator(specs, cfg, class_lambda={"micro_jet": 1e-1})
    u2 = al2.allocate(w, r, d, al2.scale)
    assert u2[cls_mask(specs, "lift_pod")].sum() >= 0.97 * 1500


def test_pure_torque_uses_micro_jets(alloc, specs, cfg):
    r, d = nominal_geometry(specs, cfg)
    A = alloc.matrix(r, d)
    jet = cls_mask(specs, "micro_jet")
    for w in ([0, 0, 0, 5.0, 0, 0], [0, 0, 0, 0, -6.0, 0], [0, 0, 0, 0, 0, 4.0],
              [0, 0, 0, 3.0, 3.0, -2.0]):
        w = np.asarray(w)
        u = alloc.allocate(w, r, d, alloc.scale)
        assert alloc.last_residual < 1e-3
        assert u[~jet].max() < 1e-3          # pods / fans ~ 0
        assert u[jet].sum() > 0


def test_torque_with_lift_uses_jets_for_torque(alloc, specs, cfg):
    """Hover-ish lift + a roll torque: the extra torque is mostly produced by jets."""
    r, d = nominal_geometry(specs, cfg)
    A = alloc.matrix(r, d)
    base = alloc.allocate([0, 0, 900.0, 0, 0, 0], r, d, alloc.scale)
    u = alloc.allocate([0, 0, 900.0, 4.0, 0, 0], r, d, alloc.scale)
    assert alloc.last_residual < 1e-3
    pods = cls_mask(specs, "lift_pod")
    dpod = np.abs(u[pods] - base[pods]).sum()
    assert dpod < 0.5 * u[pods].sum()        # pods are not re-balanced wholesale


# ----------------------------------------------------------------------------- (4) bounds
def test_bounds_and_zero_umax(alloc, specs, cfg):
    rng = np.random.default_rng(7)
    n = len(specs)
    for k in range(50):
        R = Rotation.random(random_state=k).as_matrix()
        r, d = nominal_geometry(specs, cfg, rotation=R)
        u_max = alloc.scale * rng.uniform(0.2, 1.0, n)
        u_max[rng.random(n) < 0.3] = 0.0
        if k == 0:
            u_max[cls_mask(specs, "lift_pod")] = 0.0     # pod gas exhausted
        w = rng.normal(size=6) * np.array([800, 800, 1500, 50, 50, 50])   # often infeasible
        u = alloc.allocate(w, r, d, u_max)
        assert np.all(u >= 0) and np.all(u <= u_max + 1e-9)
        assert np.all(u[u_max == 0] == 0)
        assert np.isfinite(alloc.last_residual)
    # all disabled -> zeros
    r, d = nominal_geometry(specs, cfg)
    assert not alloc.allocate([0, 0, 100, 0, 0, 0], r, d, np.zeros(n)).any()
    assert alloc.last_residual == pytest.approx(1.0)
    assert not alloc.allocate(np.zeros(6), r, d, alloc.scale).any()


# ----------------------------------------------------------------------------- (5) infeasible
def test_infeasible_downward_force(alloc, specs, cfg):
    r, d = nominal_geometry(specs, cfg)
    A = alloc.matrix(r, d)
    assert A[2].min() >= -1e-12                # nothing pushes down when upright
    u = alloc.allocate([0, 0, -500.0, 0, 0, 0], r, d, alloc.scale)
    assert np.all(u >= 0) and np.all(u <= alloc.scale + 1e-9)
    assert (A @ u)[2] <= 1e-6                  # never pushes the wrong way
    assert 0.99 <= alloc.last_residual <= 1.0 + 1e-9
    # too much lift: saturates, best effort bounded
    u = alloc.allocate([0, 0, 1e5, 0, 0, 0], r, d, alloc.scale)
    assert np.all(u <= alloc.scale + 1e-9)
    assert (A @ u)[2] > 0.95 * (A[2] * alloc.scale).clip(0).sum()


# ----------------------------------------------------------------------------- (6) timing
def test_timing(alloc, specs, cfg):
    rng = np.random.default_rng(3)
    n = len(specs)
    cases = []
    for k in range(100):
        R = Rotation.random(random_state=1000 + k).as_matrix()
        r, d = nominal_geometry(specs, cfg, rotation=R)
        cases.append((alloc.matrix(r, d) @ (rng.uniform(0, 1, n) * alloc.scale), r, d))
    alloc.allocate(*cases[0], alloc.scale)
    t0 = time.perf_counter()
    for w, r, d in cases:
        alloc.allocate(w, r, d, alloc.scale)
    mean_ms = (time.perf_counter() - t0) / len(cases) * 1e3
    print(f"\nallocate mean {mean_ms:.3f} ms/call")
    assert mean_ms < 3.0


# ----------------------------------------------------------------------------- report
def test_rank_and_torque_authority(alloc, specs, cfg):
    r, d = nominal_geometry(specs, cfg)
    A = alloc.matrix(r, d)
    assert np.linalg.matrix_rank(A) == 6
    g = cfg["sim"]["gravity"]
    m = cfg["human"]["mass"] + sum(v["total"] for v in cfg["suit"]["masses"].values())
    print(f"\nrank(A) nominal = {np.linalg.matrix_rank(A)}; sv = "
          f"{np.round(np.linalg.svd(A, compute_uv=False), 3)}")
    for label, base in (("pure", np.zeros(6)), ("hover", np.array([0, 0, m * g, 0, 0, 0]))):
        for ax, nm in zip(range(3, 6), ("Tx", "Ty", "Tz")):
            lo, hi = axis_authority(A, alloc.scale, ax, base)
            print(f"  {label:5s} {nm}: [{lo:8.2f}, {hi:8.2f}] N*m")
            assert lo < 0 < hi
