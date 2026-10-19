"""Hardware sizing model: mission requirements -> component specs -> mass budget.

All assumptions come from ``configs/sizing.yaml``.  A value written there as a 3-list
``[opt, nom, cons]`` is an assumption range; the whole model is evaluated once per case
(``CASES``) so every output carries an optimistic / nominal / conservative spread.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

from . import physics as ph

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "sizing.yaml"
CASES = ("opt", "nom", "cons")
BAR = 1.0e5


# ----------------------------------------------------------------------------- helpers
def load_sizing_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as fh:
        return yaml.safe_load(fh)


def pick(value, case: str | int = "nom"):
    """Resolve an assumption range ``[opt, nom, cons]`` for ``case``; scalars pass through."""
    idx = CASES.index(case) if isinstance(case, str) else int(case)
    if isinstance(value, (list, tuple)) and len(value) == 3 and all(
            isinstance(v, (int, float)) for v in value):
        return float(value[idx])
    return value


def _item(subsystem: str, item: str, mass: float, note: str = "") -> dict:
    return {"subsystem": subsystem, "item": item, "mass": float(mass), "note": note}


# ----------------------------------------------------------------------------- sim usage
def load_sim_usage(cfg: dict, root: Path = REPO_ROOT) -> dict:
    """Per-task pod / jet impulse and fan energy stats from results/<task>_summary.json."""
    req = cfg["mission"]["requirement"]
    out = {}
    for task in req["tasks"]:
        f = Path(root) / req["results_dir"] / f"{task}_summary.json"
        if not f.exists():
            continue
        stats = json.loads(f.read_text()).get("stats", {})
        row = {}
        for key, name in (("impulse_gas_pods", "pods"), ("impulse_gas_jets", "jets"),
                          ("battery_wh", "battery_wh")):
            if key in stats:
                row[name] = {k: float(stats[key][k]) for k in ("mean", "p95", "max")}
        row["n"] = int(stats.get("impulse_gas_pods", {}).get("n", 0))
        out[task] = row
    return out


def requirements(cfg: dict, case: str, system_mass: float, usage: dict) -> dict:
    """Pod and jet impulse requirements (sim-measured and first-principles) at ``system_mass``."""
    mis = cfg["mission"]
    rq = mis["requirement"]
    g = cfg["constants"]["g"]
    stat = rq["statistic"]
    scale = system_mass / mis["sim_system_mass"] if rq.get("scale_with_mass", True) else 1.0
    margin = pick(rq["margin"], case)

    if usage:
        pod_task = max(usage, key=lambda t: usage[t].get("pods", {}).get(stat, 0.0))
        jet_task = max(usage, key=lambda t: usage[t].get("jets", {}).get(stat, 0.0))
        sim_pod = usage[pod_task]["pods"][stat]
        sim_jet = usage[jet_task]["jets"][stat]
    else:
        pod_task = jet_task = "fallback"
        sim_pod = rq["fallback"]["pod_impulse"]
        sim_jet = rq["fallback"]["jet_impulse"]

    b = mis["braking"]
    fp = ph.braking_impulse(mis["design_drop_height"], system_mass, b["v_t"], b["h_t"], b["a"], g)
    overhead = pick(mis["sim_overhead_vs_ideal"], case)
    fp_pod = fp["I"] * overhead
    pod_basis = max(sim_pod * scale, fp_pod)
    return {
        "system_mass": system_mass, "mass_scale": scale, "margin": margin, "statistic": stat,
        "sim_pod": sim_pod, "sim_pod_task": pod_task, "sim_jet": sim_jet, "sim_jet_task": jet_task,
        "fp_braking_ideal": fp["I"], "fp_overhead": overhead, "fp_pod": fp_pod,
        "brake_thrust": fp["thrust_brake"],
        "pod_basis": "first-principles I*(H)" if fp_pod >= sim_pod * scale else f"sim {pod_task}",
        "pod_impulse": margin * pod_basis,
        "jet_impulse": margin * sim_jet * scale,
    }


def braking_table(cfg: dict, system_mass: float) -> list[dict]:
    mis, g = cfg["mission"], cfg["constants"]["g"]
    b = mis["braking"]
    rows = []
    for h in mis["drop_heights"]:
        r = ph.braking_impulse(h, system_mass, b["v_t"], b["h_t"], b["a"], g)
        rows.append({"H": h, **r})
    return rows


def jump_estimate(cfg: dict, system_mass: float) -> dict:
    mis, g = cfg["mission"], cfg["constants"]["g"]
    j = mis["jump"]
    up = ph.jump_impulse(system_mass, j["sole_rise"], j["thrust"], g, j["leg_push_credit"])
    b = mis["braking"]
    down = ph.braking_impulse(j["apex_above_platform"], system_mass, b["v_t"], b["h_t"], b["a"], g)
    return {**up, "I_landing": down["I"], "I_total": up["I"] + down["I"]}


def mvp_requirement(cfg: dict, case: str, system_mass: float, n_pods: int, height=None) -> dict:
    """Fall-protection MVP: brake a fall from ``height`` to ``impact_speed`` with one solid pulse
    (no constant-speed hold: h_t = 0)."""
    mvp = cfg["architectures"]["D"]["mvp"]
    g = cfg["constants"]["g"]
    h = mvp["drop_height"] if height is None else height
    a = pick(mvp["decel"], case)
    r = ph.braking_impulse(h, system_mass, mvp["impact_speed"], 0.0, a, g)
    margin = pick(mvp["margin"], case)
    return {"H": h, "decel": a, "ideal": r["I"], "margin": margin, "impulse": r["I"] * margin,
            "thrust_total": r["thrust_brake"], "thrust_per_pod": r["thrust_brake"] / n_pods,
            "fire_height": r["h_b"], "burn_time": r["t_brake"], "v0": r["v0"],
            "impact_v_no_thrust": r["impact_v_no_thrust"]}


# ----------------------------------------------------------------------------- storage
def size_store(cfg: dict, prop_key: str, impulse: float, case: str, sub: str,
               peak_thrust: float | None = None) -> dict:
    """Propellant + storage for ``impulse`` [N*s] of propellant ``prop_key``.
    Returns dict(items, propellant_mass, details)."""
    g0 = cfg["constants"]["g0"]
    p = cfg["propellants"][prop_key]
    st = cfg["storage"]
    kind = p["kind"]
    items, det = [], {"kind": kind, "label": p["label"], "impulse": impulse}

    if kind == "cold_gas":
        s = st["cold_gas"]
        gas = cfg["gases"]["n2"]
        isp = pick(p["isp"], case)
        m_p = ph.propellant_mass(impulse, isp, g0)
        m_load = m_p / pick(s["usable_fraction"], case)
        P = s["pressure_bar"] * BAR
        rho = ph.gas_density(P, s["temperature_K"], gas["R"], pick(s["z"], case))
        V = m_load / rho
        m_tank = ph.tank_mass(P, V, pick(s["performance_factor_m"], case),
                              pick(s["safety_factor"], case), g0)
        items += [_item(sub, "N2 gas load", m_load, f"{m_p:.2f} kg usable, Isp {isp:.0f} s"),
                  _item(sub, "Type-IV tank shell", m_tank,
                        f"{V * 1e3:.2f} L @ {s['pressure_bar']} bar"),
                  _item(sub, "tank valve / burst disc / fittings", pick(s["fixed_mass"], case))]
        det.update(isp=isp, volume_L=V * 1e3, density=rho, pressure_bar=s["pressure_bar"])

    elif kind == "liquefied_gas":
        s = st["liquefied_co2"]
        isp = pick(p["isp"], case)
        m_p = ph.propellant_mass(impulse, isp, g0)
        m_load = m_p / pick(s["usable_fraction"], case)
        V = m_load / s["fill_density"]
        m_tank = ph.tank_mass(s["pressure_bar"] * BAR, V, pick(s["performance_factor_m"], case),
                              pick(s["safety_factor"], case), g0)
        items += [_item(sub, "liquid CO2 load", m_load, f"Isp {isp:.0f} s"),
                  _item(sub, "CO2 tank shell", m_tank, f"{V * 1e3:.2f} L @ {s['pressure_bar']} bar"),
                  _item(sub, "tank valve / fittings", pick(s["fixed_mass"], case))]
        det.update(isp=isp, volume_L=V * 1e3)
        if peak_thrust:
            det["boil_power_W"] = ph.mass_flow(peak_thrust, isp, g0) * s["h_vap"]

    elif kind == "solid":
        isp = pick(p["isp"], case)
        m_p = ph.propellant_mass(impulse, isp, g0)
        pmf = pick(st["solid"]["propellant_mass_fraction"], case)
        items += [_item(sub, "solid propellant", m_p, f"Isp {isp:.0f} s"),
                  _item(sub, "motor case / nozzle / igniter (inert)", m_p * (1.0 / pmf - 1.0),
                        f"propellant mass fraction {pmf:.2f}")]
        det.update(isp=isp, pmf=pmf)

    elif kind == "monoprop":
        s = st["h2o2"]
        gas = cfg["gases"]["n2"]
        isp = pick(p["isp"], case)
        m_p = ph.propellant_mass(impulse, isp, g0)
        v_liq = m_p / s["density"]
        v_tank = v_liq * (1.0 + pick(s["ullage"], case))
        p_t = pick(s["tank_pressure_bar"], case) * BAR
        p_b = s["pressurant_bottle_bar"] * BAR
        z_b = pick(s["pressurant_z"], case)
        m_tank = ph.tank_mass(p_t, v_tank, pick(s["performance_factor_m"], case),
                              pick(s["safety_factor"], case), g0)
        v_b = ph.blowdown_pressurant_volume(p_t, v_tank, p_b, z_b)
        T = cfg["storage"]["cold_gas"]["temperature_K"]
        m_press = ph.gas_density(p_b, T, gas["R"], z_b) * v_b
        cg = st["cold_gas"]
        m_bottle = ph.tank_mass(p_b, v_b, pick(cg["performance_factor_m"], case),
                                pick(cg["safety_factor"], case), g0)
        items += [_item(sub, "90 % H2O2", m_p, f"Isp {isp:.0f} s, {v_liq * 1e3:.2f} L"),
                  _item(sub, "peroxide tank", m_tank,
                        f"{v_tank * 1e3:.2f} L @ {p_t / BAR:.0f} bar (Al/lined)"),
                  _item(sub, "N2 pressurant gas", m_press, f"{v_b * 1e3:.2f} L @ 300 bar"),
                  _item(sub, "pressurant bottle", m_bottle),
                  _item(sub, "regulator / vent / check / burst disc", pick(s["fixed_mass"], case))]
        det.update(isp=isp, volume_L=v_tank * 1e3, pressurant_L=v_b * 1e3)

    elif kind == "airbreathing":
        tsfc = pick(p["tsfc"], case)
        isp = ph.tsfc_to_isp(tsfc, g0)
        m_fuel = ph.propellant_mass(impulse, isp, g0)
        items += [_item(sub, "kerosene fuel", m_fuel, f"TSFC {tsfc} kg/(N h) = Isp {isp:.0f} s"),
                  _item(sub, "fuel tank + pump", 0.3 + 0.2 * m_fuel)]
        if peak_thrust:
            tw = pick(p["engine_thrust_to_weight"], case)
            m_eng = peak_thrust / (tw * g0)
            items.append(_item(sub, "turbojet engines", m_eng,
                               f"{peak_thrust:.0f} N at T/W {tw:.0f}"))
        m_p = m_fuel
        det.update(isp=isp, spool_up_s=pick(p["spool_up_s"], case))
    else:
        raise ValueError(f"unknown propellant kind {kind!r}")

    det["propellant_mass"] = m_p
    det["store_mass"] = sum(i["mass"] for i in items)
    return {"items": items, "propellant_mass": m_p, "details": det}


# ----------------------------------------------------------------------------- subsystems
def size_pods(cfg: dict, pods: dict, impulse: float, case: str) -> dict:
    prop = pods["propellant"]
    kind = cfg["propellants"][prop]["kind"]
    n = int(pods["n"])
    store = size_store(cfg, prop, impulse, case, "pods")
    hw = pick(cfg["thrusters"]["pods"]["hardware_mass"][kind], case)
    items = store["items"] + [_item("pods", f"pod thruster hardware x{n}", n * hw)]
    return {"items": items, "propellant_mass": store["propellant_mass"],
            "details": store["details"]}


def size_jets(cfg: dict, jets: dict, impulse: float, case: str) -> dict:
    j = cfg["thrusters"]["micro_jet"]
    pl = cfg["thrusters"]["plumbing"]
    n = int(jets["n"])
    store = size_store(cfg, jets["propellant"], impulse, case, "jets")
    tube = n * (pick(pl["tube_length_per_jet"], case) * pick(pl["tube_mass_per_m"], case)
                + pick(pl["fitting_mass_per_jet"], case))
    items = store["items"] + [
        _item("jets", f"solenoid valve + nozzle x{n}", n * pick(j["valve_nozzle_mass"], case),
              f"{pick(j['valve_nozzle_mass'], case) * 1e3:.0f} g each"),
        _item("jets", f"valve drivers + wiring x{n}", n * pick(j["driver_wiring_mass"], case)),
        _item("jets", "feed tubing + fittings", tube),
        _item("jets", "manifolds", pick(pl["manifold_mass"], case)),
        _item("jets", "regulator", pick(pl["regulator_mass"], case)),
    ]
    return {"items": items, "propellant_mass": store["propellant_mass"],
            "details": store["details"]}


def fan_point(cfg: dict, case: str, diameter: float | None = None) -> dict:
    f = cfg["fans"]
    rho = cfg["constants"]["rho_air"]
    d = pick(f["duct_diameter"], case) if diameter is None else diameter
    fom = pick(f["figure_of_merit"], case)
    T = f["thrust"]
    area = ph.disk_area(d)
    p_ideal = ph.ideal_induced_power(T, area, rho)
    p_shaft = p_ideal / fom
    p_elec = p_shaft / pick(f["motor_esc_efficiency"], case)
    spec = cfg["mission"]["spec_budgets"]["fan_peak_power_w"]
    a_spec = ph.area_for_power(T, spec, fom, rho)
    return {"diameter": d, "area": area, "fom": fom, "p_ideal": p_ideal, "p_shaft": p_shaft,
            "p_elec": p_elec, "ratio_to_spec": p_shaft / spec,
            "spec_power": spec, "diameter_for_spec_power": ph.diameter_from_area(a_spec),
            "induced_velocity": math.sqrt(T / (2 * rho * area))}


def size_fans(cfg: dict, case: str) -> dict:
    f = cfg["fans"]
    n = int(f["n"])
    fp = fan_point(cfg, case)
    m_motor = fp["p_shaft"] / pick(f["motor_specific_power"], case)
    e_wh = cfg["mission"]["spec_budgets"]["battery_wh"]
    p_tot = n * fp["p_elec"]
    m_batt_e = e_wh / pick(f["battery_specific_energy"], case)
    m_batt_p = p_tot / pick(f["battery_specific_power"], case)
    m_batt = max(m_batt_e, m_batt_p)
    items = [
        _item("fans", f"motor + ESC x{n}", n * m_motor, f"{fp['p_shaft'] / 1e3:.1f} kW shaft each"),
        _item("fans", f"rotor + duct x{n}", n * pick(f["fan_duct_mass"], case),
              f"{fp['diameter'] * 100:.0f} cm duct"),
        _item("fans", "fan battery", m_batt,
              "power-limited" if m_batt_p > m_batt_e else "energy-limited"),
    ]
    energy = m_batt * pick(f["battery_specific_energy"], case)
    return {"items": items, "point": fp, "power_total_elec": p_tot,
            "battery_mass_energy": m_batt_e, "battery_mass_power": m_batt_p,
            "full_power_endurance_s": energy * 3600.0 / p_tot,
            "spec_c_rate": p_tot / e_wh}


def size_other(cfg: dict, arch: dict, case: str) -> list[dict]:
    over = arch.get("other_override") or {}
    items = []
    for key in arch.get("other", []):
        val = over.get(key, cfg["other_mass"][key])
        items.append(_item("structure/other", key, pick(val, case)))
    for key, val in over.items():
        if key not in arch.get("other", []):
            items.append(_item("structure/other", key, pick(val, case)))
    return items


# ----------------------------------------------------------------------------- architectures
def _impulse(spec, kind: str, req: dict, cfg: dict) -> float:
    if isinstance(spec, (int, float)):
        return float(spec)
    if spec == "spec":
        return cfg["mission"]["spec_budgets"][f"{kind}_impulse"]
    if spec == "required":
        return req[f"{kind}_impulse"]
    raise ValueError(f"unknown impulse spec {spec!r}")


def size_architecture(cfg: dict, key: str, case: str = "nom", usage: dict | None = None,
                      iterations: int = 50) -> dict:
    """Size one architecture; system mass (human + suit) is iterated to a fixed point because
    the impulse requirement grows with the suit's own mass."""
    arch = cfg["architectures"][key]
    mis = cfg["mission"]
    usage = load_sim_usage(cfg) if usage is None else usage
    g = cfg["constants"]["g"]
    suit = mis["spec_suit_mass"]
    for _ in range(iterations):
        m_sys = mis["human_mass"] + suit
        req = requirements(cfg, case, m_sys, usage)
        items, pods_res, jets_res, fans_res, mvp = [], None, None, None, None
        pod_thrust_each = cfg["thrusters"]["pods"]["thrust"]
        if arch.get("pods"):
            pods = arch["pods"]
            if pods["impulse"] == "mvp":
                mvp = mvp_requirement(cfg, case, m_sys, int(pods["n"]))
                pod_imp = mvp["impulse"]
                pod_thrust_each = mvp["thrust_per_pod"]
            else:
                pod_imp = _impulse(pods["impulse"], "pod", req, cfg)
            pods_res = size_pods(cfg, pods, pod_imp, case)
            pods_res["impulse"] = pod_imp
            items += pods_res["items"]
        if arch.get("jets"):
            jet_imp = _impulse(arch["jets"]["impulse"], "jet", req, cfg)
            jets_res = size_jets(cfg, arch["jets"], jet_imp, case)
            jets_res["impulse"] = jet_imp
            items += jets_res["items"]
        if arch.get("fans"):
            fans_res = size_fans(cfg, case)
            items += fans_res["items"]
        items += size_other(cfg, arch, case)
        new_suit = sum(i["mass"] for i in items)
        converged = abs(new_suit - suit) < 1e-9
        suit = new_suit
        if converged:
            break
    m_sys = mis["human_mass"] + suit
    n_pods = int(arch["pods"]["n"]) if arch.get("pods") else 0
    thrust_up = n_pods * pod_thrust_each
    if fans_res:
        thrust_up += cfg["fans"]["n"] * cfg["fans"]["thrust"]
    subsystems: dict[str, float] = {}
    for i in items:
        subsystems[i["subsystem"]] = subsystems.get(i["subsystem"], 0.0) + i["mass"]
    prop = sum(r["propellant_mass"] for r in (pods_res, jets_res) if r)
    res = {
        "key": key, "name": arch["name"], "case": case, "items": items,
        "subsystems": subsystems, "suit_mass": suit, "system_mass": m_sys,
        "propellant_mass": prop, "requirements": req, "pods": pods_res, "jets": jets_res,
        "fans": fans_res, "mvp": mvp, "peak_vertical_thrust": thrust_up,
        "thrust_to_weight": thrust_up / (m_sys * g),
        "n_jets": int(arch["jets"]["n"]) if arch.get("jets") else 0,
    }
    return res


def verdict(cfg: dict, res_by_case: dict) -> dict:
    """Feasibility verdict from the three cases of one architecture."""
    nom, opt = res_by_case["nom"], res_by_case["opt"]
    lim = cfg["mission"]["wearable_mass_limit"]
    blocking, caveats = [], []
    if nom["fans"]:
        fp = nom["fans"]["point"]
        if fp["ratio_to_spec"] > 3.0:
            blocking.append(
                f"fans: {fp['p_shaft'] / 1e3:.0f} kW shaft per 300 N fan "
                f"(range {opt['fans']['point']['p_shaft'] / 1e3:.0f}-"
                f"{res_by_case['cons']['fans']['point']['p_shaft'] / 1e3:.0f} kW) vs "
                f"{fp['spec_power'] / 1e3:.1f} kW in the spec; fan system "
                f"{nom['subsystems']['fans']:.1f} kg; 1.2 kW would need a "
                f"{fp['diameter_for_spec_power']:.1f} m rotor")
    if opt["suit_mass"] > lim:
        blocking.append(f"suit mass {opt['suit_mass']:.1f} kg even optimistically "
                        f"(> {lim:.0f} kg wearable limit)")
    elif nom["suit_mass"] > lim:
        caveats.append(f"suit mass {nom['suit_mass']:.1f} kg nominal > {lim:.0f} kg target")
    if nom["suit_mass"] > 1.3 * cfg["mission"]["spec_suit_mass"]:
        caveats.append(f"suit is {nom['suit_mass']:.1f} kg, not the simulated "
                       f"{cfg['mission']['spec_suit_mass']:.0f} kg: re-run the sim at this mass")
    if nom["pods"]:
        kind = nom["pods"]["details"]["kind"]
        if kind == "solid":
            caveats.append("solid pods cannot throttle: the sim's continuous 0-1500 N, 5 ms pods "
                           "must become discrete pulses (controller rework)")
        if kind == "monoprop":
            caveats.append("H2O2 pods: 10-50 ms response (sim assumes 5 ms); 90 % peroxide is a "
                           "restricted, hazardous oxidizer")
    need = nom["requirements"]["brake_thrust"] if not nom["mvp"] else nom["mvp"]["thrust_total"]
    if nom["peak_vertical_thrust"] + 1e-6 < need:
        blocking.append(f"peak lift {nom['peak_vertical_thrust']:.0f} N < braking need {need:.0f} N")
    if nom["mvp"]:
        caveats.append("no attitude control: relies on near-upright falls; airbags cover the rest")
    status = "INFEASIBLE" if blocking else ("FEASIBLE with caveats" if caveats else "FEASIBLE")
    return {"status": status, "blocking": blocking, "caveats": caveats}


# ----------------------------------------------------------------------------- comparisons
def propellant_comparison(cfg: dict, pod_impulse: float, jet_impulse: float,
                          pod_peak_thrust: float, jet_peak_thrust: float) -> list[dict]:
    """Propellant + storage mass of each option for the pod and the jet requirement."""
    rows = []
    for key, p in cfg["propellants"].items():
        row = {"key": key, "label": p["label"], "exhaust": p["exhaust_temp_C"],
               "response": p["response"]}
        for which, imp, thr in (("pod", pod_impulse, pod_peak_thrust),
                                ("jet", jet_impulse, jet_peak_thrust)):
            vals = []
            for c in CASES:
                s = size_store(cfg, key, imp, c, which, peak_thrust=thr)
                vals.append((s["propellant_mass"], s["details"]["store_mass"], s["details"]))
            row[which] = {"propellant": [v[0] for v in vals], "total": [v[1] for v in vals],
                          "isp": [v[2]["isp"] for v in vals],
                          "volume_L": [v[2].get("volume_L") for v in vals],
                          "boil_power_W": [v[2].get("boil_power_W") for v in vals]}
        rows.append(row)
    # electric ducted fans for the same peak lift as the pods (power-limited battery + motors)
    fans_row = {"key": "edf", "label": "electric ducted fans (same peak lift as pods)",
                "exhaust": "ambient air, 100-115 dB", "response": "50-150 ms spool"}
    totals = []
    for c in CASES:
        fp = fan_point(cfg, c)
        n = pod_peak_thrust / cfg["fans"]["thrust"]
        f = cfg["fans"]
        m = n * (fp["p_shaft"] / pick(f["motor_specific_power"], c) + pick(f["fan_duct_mass"], c)) \
            + n * fp["p_elec"] / pick(f["battery_specific_power"], c)
        totals.append(m)
    fans_row["pod"] = {"propellant": [0.0] * 3, "total": totals, "isp": [None] * 3,
                       "volume_L": [None] * 3, "boil_power_W": [None] * 3}
    fans_row["jet"] = None
    rows.append(fans_row)
    return rows


def nozzle_table(cfg: dict, n_jets: int, case: str = "nom") -> list[dict]:
    j = cfg["thrusters"]["micro_jet"]
    gas = cfg["gases"]["n2"]
    g0 = cfg["constants"]["g0"]
    F = j["thrust"]
    cf = pick(j["thrust_coefficient"], case)
    T = pick(j["gas_temperature_K"], case)
    isp = pick(cfg["propellants"]["n2_cold_gas"]["isp"], case)
    cs = ph.c_star(gas["gamma"], gas["R"], T)
    rows = []
    for pc_bar in j["chamber_pressure_sweep_bar"]:
        pc = pc_bar * BAR
        at = ph.throat_area(F, cf, pc)
        mdot_isp = ph.mass_flow(F, isp, g0)
        mdot_cs = pc * at / cs
        rows.append({
            "pc_bar": pc_bar, "throat_mm2": at * 1e6, "throat_d_mm": ph.diameter_from_area(at) * 1e3,
            "mdot_isp": mdot_isp, "mdot_cstar": mdot_cs, "isp_cstar": cf * cs / g0,
            "exit_T_C": ph.isentropic_temperature(T, 1.0 / pc_bar, gas["gamma"]) - 273.15,
            "peak_flow_all": mdot_isp * n_jets * pick(j["peak_simultaneous_frac"], case),
        })
    return rows


def jet_valve_summary(cfg: dict, case: str = "nom") -> dict:
    j = cfg["thrusters"]["micro_jet"]
    g0 = cfg["constants"]["g0"]
    isp = pick(cfg["propellants"]["n2_cold_gas"]["isp"], case)
    t_on = pick(j["min_on_time_ms"], case) * 1e-3
    pc = pick(j["chamber_pressure_bar"], case) * BAR
    at = ph.throat_area(j["thrust"], pick(j["thrust_coefficient"], case), pc)
    return {"thrust": j["thrust"], "pc_bar": pc / BAR, "throat_d_mm": ph.diameter_from_area(at) * 1e3,
            "mdot": ph.mass_flow(j["thrust"], isp, g0),
            "response_ms": pick(j["valve_response_ms"], case), "min_on_ms": t_on * 1e3,
            "min_impulse_bit": j["thrust"] * t_on, "valve_mass_g": pick(j["valve_nozzle_mass"], case) * 1e3}


def fan_sweep(cfg: dict) -> list[dict]:
    rows = []
    for d in cfg["fans"]["diameter_sweep"]:
        pts = [fan_point(cfg, c, diameter=d) for c in CASES]
        rows.append({"diameter": d, "p_ideal": pts[1]["p_ideal"],
                     "p_shaft": [p["p_shaft"] for p in pts], "v_induced": pts[1]["induced_velocity"],
                     "ratio_to_spec": pts[1]["ratio_to_spec"]})
    return rows


def allocator_estimate(cfg: dict, n_thrusters: int, measured_ms: float | None = None) -> dict:
    e = cfg["embedded"]
    m = e["n_wrench_rows"]
    budget = e["control_period_ms"] * e["allocator_budget_frac"]
    flops = []
    for c in CASES:
        it = pick(e["active_set_iterations"], c)
        flops.append(2 * m * n_thrusters ** 2 + it * (n_thrusters ** 3 / 3 + 4 * n_thrusters ** 2))
    plats = []
    for name, mflops in e["platforms"].items():
        t = [flops[i] / (pick(mflops, c) * 1e6) * 1e3 for i, c in enumerate(CASES)]
        status = ("OK" if t[2] <= budget else "OK nominal" if t[1] <= budget
                  else "tight" if t[1] <= e["control_period_ms"] else "too slow")
        plats.append({"platform": name, "ms": t, "status": status})
    return {"n": n_thrusters, "flops": flops, "budget_ms": budget,
            "period_ms": e["control_period_ms"],
            "x86_reference_ms": e["x86_ms_per_call_reference"], "x86_measured_ms": measured_ms,
            "platforms": plats}


def bench_allocator(n_calls: int = 200, variant: str = "suit_variant_side_jets.yaml") -> dict | None:
    """Time the simulator's FastAllocator on this machine (None if it cannot be imported)."""
    try:
        import platform
        import time

        import numpy as np
        from scipy.spatial.transform import Rotation

        from suitsim.control.allocation import nominal_geometry
        from suitsim.control.fast_allocation import FastAllocator
        from suitsim.interfaces import CONFIG_DIR, load_config, load_thruster_specs
    except Exception:                                    # pragma: no cover - optional
        return None
    try:
        cfg = load_config(CONFIG_DIR / variant)
        specs = load_thruster_specs(cfg)
        al = FastAllocator(specs, cfg)
        rng = np.random.default_rng(3)
        cases = []
        # smooth 500 Hz-like trajectory (slow tumble + slowly varying wrench), so the
        # warm start sees consecutive control steps as it does inside the simulator loop
        u0 = rng.uniform(0, 1, len(specs)) * al.scale
        du = rng.normal(0, 0.01, len(specs)) * al.scale
        for k in range(n_calls):
            R = Rotation.from_rotvec([0.004 * k, 0.002 * k, 0.001 * k]).as_matrix()
            r, d = nominal_geometry(specs, cfg, rotation=R)
            u = np.clip(u0 + du * np.sin(0.05 * k), 0, al.scale)
            cases.append((al.matrix(r, d) @ u, r, d))
        al.allocate(*cases[0], al.scale)
        t0 = time.perf_counter()
        for w, r, d in cases:
            al.allocate(w, r, d, al.scale)
        return {"n_thrusters": len(specs), "ms_per_call": (time.perf_counter() - t0) / n_calls * 1e3,
                "machine": platform.processor() or platform.machine()}
    except Exception:                                    # pragma: no cover - optional
        return None


def bom(cfg: dict, n_jets: int, key: str = "bom_B") -> dict:
    """Cost list with (low, high) USD ranges; ``qty: n_jets`` scales with the jet count."""
    rows, lo, hi = [], 0.0, 0.0
    for r in cfg["costs"][key]:
        qty = n_jets if r["qty"] == "n_jets" else int(r["qty"])
        a, b = r["unit"]
        rows.append({"item": r["item"], "qty": qty, "unit": (a, b), "total": (a * qty, b * qty),
                     "week": r.get("week")})
        lo += a * qty
        hi += b * qty
    return {"rows": rows, "total": (lo, hi)}


def sim_config_crosscheck(cfg: dict, configs=("suit_default.yaml", "suit_realistic.yaml",
                                               "suit_realistic_peroxide.yaml")) -> list[dict]:
    """Suit mass and budgets the simulator configs assume (read-only), for comparison with the
    sized architectures.  Missing / unloadable configs are skipped."""
    rows = []
    try:
        from suitsim.interfaces import CONFIG_DIR, load_config
    except Exception:                                    # pragma: no cover - optional
        return rows
    g0 = cfg["constants"]["g0"]
    for name in configs:
        try:
            c = load_config(CONFIG_DIR / name)
        except Exception:                                # pragma: no cover - optional
            continue
        suit = c.get("suit", {})
        dry = sum(float(v["total"]) for v in (suit.get("masses") or {}).values() if v)
        bud = suit.get("budgets", {})
        prop = 0.0
        for k, v in (suit.get("propellant") or {}).items():
            if v and k in bud:
                prop += float(bud[k]) / (float(v["propellant_isp"]) * g0)
        cls = suit.get("actuator_classes", {})
        n_jets = sum(1 for t in (suit.get("thrusters") or {}).values()
                     if t and t.get("cls") == "micro_jet")
        rows.append({"config": name, "dry_mass": dry, "propellant_mass": prop,
                     "suit_mass": dry + prop, "pod_impulse": bud.get("impulse_gas_pods"),
                     "jet_impulse": bud.get("impulse_gas_jets"), "n_jets": n_jets,
                     "fans": bool(cls.get("fan")),
                     "pod_tau": (cls.get("lift_pod") or {}).get("tau")})
    return rows


# ----------------------------------------------------------------------------- top level
def run(cfg: dict | None = None, archs: list[str] | None = None, usage: dict | None = None,
        bench: bool = False) -> dict:
    cfg = load_sizing_config() if cfg is None else cfg
    usage = load_sim_usage(cfg) if usage is None else usage
    archs = list(cfg["architectures"]) if not archs else archs
    out = {"usage": usage, "architectures": {}}
    for k in archs:
        by_case = {c: size_architecture(cfg, k, c, usage) for c in CASES}
        out["architectures"][k] = {"cases": by_case, "verdict": verdict(cfg, by_case)}
    m_ref = cfg["mission"]["sim_system_mass"]
    out["braking_table"] = braking_table(cfg, m_ref)
    out["jump"] = jump_estimate(cfg, m_ref)
    out["requirements_at_sim_mass"] = {c: requirements(cfg, c, m_ref, usage) for c in CASES}
    req = out["requirements_at_sim_mass"]["nom"]
    out["propellants"] = propellant_comparison(
        cfg, req["pod_impulse"], req["jet_impulse"], 2 * cfg["thrusters"]["pods"]["thrust"],
        48 * cfg["thrusters"]["micro_jet"]["thrust"] * pick(
            cfg["thrusters"]["micro_jet"]["peak_simultaneous_frac"], "nom"))
    out["fan_sweep"] = fan_sweep(cfg)
    out["nozzles"] = nozzle_table(cfg, 48)
    out["jet_valve"] = jet_valve_summary(cfg)
    mvp_h = cfg["architectures"]["D"]["mvp"]["drop_heights"]
    m_d = cfg["mission"]["human_mass"] + (out["architectures"]["D"]["cases"]["nom"]["suit_mass"]
                                          if "D" in out["architectures"] else 5.0)
    out["mvp_table"] = [mvp_requirement(cfg, "nom", m_d, 2, h) for h in mvp_h]
    measured = bench_allocator() if bench else None
    out["allocator_bench"] = measured
    out["allocator"] = allocator_estimate(cfg, 48, measured["ms_per_call"] if measured else None)
    out["bom_B"] = bom(cfg, 48, "bom_B")
    out["phase1"] = bom(cfg, 48, "phase1_purchases")
    out["pod_stand"] = bom(cfg, 48, "pod_stand_purchases")
    out["sim_crosscheck"] = sim_config_crosscheck(cfg)
    return out
