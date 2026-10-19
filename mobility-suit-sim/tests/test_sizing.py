"""Hardware sizing calculator (suitsim.sizing): formulas against hand calculations, monotonic
storage scaling, and self-consistent architecture budgets."""
import json
import math

import pytest

from suitsim.sizing import CASES, load_sizing_config, pick, run, size_architecture
from suitsim.sizing import physics as ph
from suitsim.sizing.model import load_sim_usage, requirements
from suitsim.sizing.report import BEGIN, END, render, update_doc


@pytest.fixture(scope="module")
def cfg():
    return load_sizing_config()


@pytest.fixture(scope="module")
def usage(cfg):
    return load_sim_usage(cfg)


@pytest.fixture(scope="module")
def result(cfg, usage):
    return run(cfg, usage=usage)


# ----------------------------------------------------------------------------- formulas
def test_propellant_mass():
    # 2500 N*s solid at Isp 200 s: 2500 / (200 * 9.80665) = 1.2746 kg
    assert ph.propellant_mass(2500.0, 200.0) == pytest.approx(1.27465, rel=1e-4)
    # 600 N*s cold N2 at 70 s -> 0.874 kg (the figure quoted in suit_realistic.yaml)
    assert ph.propellant_mass(600.0, 70.0) == pytest.approx(0.874, abs=1e-3)
    assert ph.mass_flow(40.0, 68.0) == pytest.approx(40.0 / (68.0 * 9.80665))
    with pytest.raises(ValueError):
        ph.propellant_mass(1.0, 0.0)


def test_tsfc_to_isp():
    # 0.15 kg/(N h) = 4.167e-5 kg/(N s) -> Isp = 1 / (4.167e-5 * g0) = 2447 s
    assert ph.tsfc_to_isp(0.15) == pytest.approx(2447.3, rel=1e-3)


def test_fan_momentum_theory_hand_calc():
    # T = 300 N, D = 0.20 m: A = 0.0314159 m^2, 2 rho A = 0.0769690,
    # T^3 / (2 rho A) = 2.7e7 / 0.076969 = 3.5079e8, sqrt = 18729 W
    A = math.pi * 0.2 ** 2 / 4
    assert ph.disk_area(0.2) == pytest.approx(A)
    assert ph.ideal_induced_power(300.0, A, 1.225) == pytest.approx(18729.6, rel=1e-3)
    # with FoM 0.6-0.75 and 15-25 cm ducts the shaft power is ~20-42 kW, >> spec 1.2 kW
    for d in (0.15, 0.20, 0.25):
        for fom in (0.6, 0.75):
            p = ph.fan_shaft_power(300.0, d, fom, 1.225)
            assert 19e3 < p < 43e3
    # area_for_power inverts fan_shaft_power
    a = ph.area_for_power(300.0, 1200.0, 0.7, 1.225)
    assert ph.ideal_induced_power(300.0, a, 1.225) / 0.7 == pytest.approx(1200.0, rel=1e-9)
    assert ph.diameter_from_area(a) > 4.0                  # 1.2 kW needs a > 4 m rotor


def test_gas_density_and_tank_mass():
    # N2 at 300 bar, 300 K, Z = 1.1: 3e7 / (1.1 * 296.8 * 300) = 306.3 kg/m^3
    assert ph.gas_density(300e5, 300.0, 296.8, 1.1) == pytest.approx(306.3, rel=1e-3)
    # SCBA sanity: 6.8 L at 300 bar, SF 3, PV/W 25 km -> 3*3e7*6.8e-3/(25e3*g0) = 2.50 kg
    assert ph.tank_mass(300e5, 6.8e-3, 25e3, 3.0) == pytest.approx(2.496, rel=1e-3)


@pytest.mark.parametrize("p_bar", [100, 200, 300, 450])
def test_tank_mass_monotonic(p_bar):
    base = ph.tank_mass(p_bar * 1e5, 3e-3, 30e3, 2.5)
    assert ph.tank_mass((p_bar + 50) * 1e5, 3e-3, 30e3, 2.5) > base
    assert ph.tank_mass(p_bar * 1e5, 4e-3, 30e3, 2.5) > base
    assert ph.tank_mass(p_bar * 1e5, 3e-3, 40e3, 2.5) < base      # better material, lighter
    assert ph.tank_mass(p_bar * 1e5, 3e-3, 30e3, 3.0) > base      # more safety factor, heavier


def test_cold_gas_store_monotonic_in_impulse(cfg):
    from suitsim.sizing.model import size_store
    masses = [size_store(cfg, "n2_cold_gas", I, "nom", "jets")["details"]["store_mass"]
              for I in (300, 600, 900, 1200)]
    assert all(b > a for a, b in zip(masses, masses[1:]))


def test_nozzle_formulas():
    # A* = F / (C_F Pc) = 40 / (1.5 * 15e5) = 17.78 mm^2
    at = ph.throat_area(40.0, 1.5, 15e5)
    assert at * 1e6 == pytest.approx(17.78, rel=1e-3)
    assert ph.diameter_from_area(at) * 1e3 == pytest.approx(4.758, rel=1e-3)
    # N2 c* at 300 K ~ 432 m/s; C_F c* / g0 ~ 66 s, consistent with the 60-73 s Isp range
    cs = ph.c_star(1.4, 296.8, 300.0)
    assert cs == pytest.approx(432.0, rel=0.01)
    assert 60 < 1.5 * cs / 9.80665 < 73
    # mass flow from the throat matches F / (Isp g0) at that Isp
    assert 15e5 * at / cs == pytest.approx(ph.mass_flow(40.0, 1.5 * cs / 9.80665), rel=1e-9)
    # isentropic expansion 15 bar -> 1 bar from 300 K cools far below ambient
    assert ph.isentropic_temperature(300.0, 1 / 15, 1.4) < 160.0


def test_braking_impulse_matches_paper():
    # paper: H = 3 m, m = 100 kg -> h_b = 1.67 m, v0 = 5.11 m/s, I* = 1062 N*s
    r = ph.braking_impulse(3.0, 100.0, 1.2, 0.3, 9.0, 9.81)
    assert r["h_b"] == pytest.approx(1.67, abs=0.01)
    assert r["v0"] == pytest.approx(5.11, abs=0.01)
    assert r["I"] == pytest.approx(1062.0, abs=2.0)
    # monotonic in height and linear in mass
    Is = [ph.braking_impulse(h, 100.0, 1.2, 0.3, 9.0)["I"] for h in (0.2, 0.5, 1, 2, 3, 5)]
    assert all(b >= a for a, b in zip(Is, Is[1:]))
    assert ph.braking_impulse(4.0, 50.0, 1.2, 0.3, 9.0)["I"] == pytest.approx(
        0.5 * ph.braking_impulse(4.0, 100.0, 1.2, 0.3, 9.0)["I"])
    # a drop too short to reach v_t needs nothing
    assert ph.braking_impulse(0.05, 100.0, 1.2, 0.3, 9.0)["I"] == 0.0


def test_jump_impulse_matches_paper():
    # paper: 3.13 m rise -> 7.8 m/s, 2.9 kN on 100 kg -> ~19 m/s^2 net, ~0.41 s, ~1.2 kN*s
    j = ph.jump_impulse(100.0, 3.13, 2900.0, 9.81)
    assert j["v_takeoff"] == pytest.approx(7.84, abs=0.01)
    assert j["a_net"] == pytest.approx(19.19, abs=0.01)
    assert j["t_burn"] == pytest.approx(0.41, abs=0.01)
    assert j["I"] == pytest.approx(1184, abs=5)


# ----------------------------------------------------------------------------- model
def test_pick():
    assert pick([1, 2, 3], "opt") == 1 and pick([1, 2, 3]) == 2 and pick([1, 2, 3], "cons") == 3
    assert pick(5.0, "cons") == 5.0
    assert pick("solid") == "solid"


def test_requirements_scale_with_mass(cfg, usage):
    r100 = requirements(cfg, "nom", 100.0, usage)
    r120 = requirements(cfg, "nom", 120.0, usage)
    assert r120["pod_impulse"] == pytest.approx(1.2 * r100["pod_impulse"], rel=1e-9)
    assert r120["jet_impulse"] == pytest.approx(1.2 * r100["jet_impulse"], rel=1e-9)
    # margin is applied on top of the larger of sim use and first-principles
    assert r100["pod_impulse"] >= r100["margin"] * r100["sim_pod"] - 1e-9
    assert r100["pod_impulse"] >= r100["margin"] * r100["fp_pod"] - 1e-9


def test_requirements_fallback_without_results(cfg):
    r = requirements(cfg, "nom", 100.0, {})
    assert r["sim_pod"] == cfg["mission"]["requirement"]["fallback"]["pod_impulse"]
    assert r["jet_impulse"] > 0


@pytest.mark.parametrize("key", ["A", "B", "C", "D"])
def test_architecture_totals_sum(cfg, usage, key):
    for case in CASES:
        res = size_architecture(cfg, key, case, usage)
        total = sum(i["mass"] for i in res["items"])
        assert res["suit_mass"] == pytest.approx(total, rel=1e-12)
        assert sum(res["subsystems"].values()) == pytest.approx(total, rel=1e-12)
        assert res["system_mass"] == pytest.approx(cfg["mission"]["human_mass"] + total)
        assert all(i["mass"] >= 0 for i in res["items"])
        prop = sum(r["propellant_mass"] for r in (res["pods"], res["jets"]) if r)
        assert res["propellant_mass"] == pytest.approx(prop)


@pytest.mark.parametrize("key", ["B", "C"])
def test_fixed_point_requirement_consistent(cfg, usage, key):
    """The 'required' impulses are those of the converged system mass."""
    res = size_architecture(cfg, key, "nom", usage)
    req = requirements(cfg, "nom", res["system_mass"], usage)
    assert res["pods"]["impulse"] == pytest.approx(req["pod_impulse"], rel=1e-6)
    assert res["jets"]["impulse"] == pytest.approx(req["jet_impulse"], rel=1e-6)


def test_case_ordering(result):
    for key, a in result["architectures"].items():
        m = [a["cases"][c]["suit_mass"] for c in CASES]
        assert m[0] <= m[1] <= m[2], key


def test_verdicts(result, cfg):
    A = result["architectures"]
    assert A["A"]["verdict"]["status"] == "INFEASIBLE"
    assert any("fans" in b for b in A["A"]["verdict"]["blocking"])
    for k in ("B", "C", "D"):
        assert A[k]["verdict"]["status"] != "INFEASIBLE", A[k]["verdict"]
    # B and C below the wearable target, D far lighter, all with T/W > 1.5
    assert A["B"]["cases"]["nom"]["suit_mass"] < cfg["mission"]["wearable_mass_limit"]
    assert A["D"]["cases"]["nom"]["suit_mass"] < A["B"]["cases"]["nom"]["suit_mass"]
    for k in A:
        assert A[k]["cases"]["nom"]["thrust_to_weight"] > 1.5
    # fans in A: tens of kW per fan vs 1.2 kW in the spec
    fp = A["A"]["cases"]["nom"]["fans"]["point"]
    assert 15e3 < fp["p_shaft"] < 45e3


def test_mvp_2m(result):
    m = result["architectures"]["D"]["cases"]["nom"]["mvp"]
    assert m["H"] == 2.0
    assert m["ideal"] > 0 and m["impulse"] > m["ideal"]
    # Delta-v from v0 to 2 m/s with gravity loss: I = m (v0 - 2)(1 + g/a)
    sysm = result["architectures"]["D"]["cases"]["nom"]["system_mass"]
    assert m["ideal"] == pytest.approx(sysm * (m["v0"] - 2.0) * (1 + 9.81 / m["decel"]), rel=1e-9)


def test_render_and_doc(result, cfg, tmp_path):
    txt = render(result, cfg)
    md = render(result, cfg, md=True)
    assert "INFEASIBLE" in txt and "| arch |" in md
    doc = tmp_path / "doc.md"
    doc.write_text(f"intro\n{BEGIN}\nSTALE-TEXT\n{END}\noutro\n")
    assert update_doc(doc, md)
    t = doc.read_text()
    assert "STALE-TEXT" not in t and t.startswith("intro") and t.rstrip().endswith("outro")


def test_script_json(tmp_path):
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "scripts" / "size_suit.py"
    spec = importlib.util.spec_from_file_location("size_suit", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "s.json"
    assert mod.main(["--quiet", "--json", str(out), "--arch", "B"]) == 0
    d = json.loads(out.read_text())
    assert set(d["architectures"]) == {"A", "B", "C", "D"}
