"""Text / Markdown rendering of :func:`suitsim.sizing.model.run` results."""
from __future__ import annotations

from .model import CASES

BEGIN = "<!-- BEGIN GENERATED: scripts/size_suit.py -->"
END = "<!-- END GENERATED: scripts/size_suit.py -->"


def table(headers: list[str], rows: list[list], md: bool) -> str:
    rows = [[str(c) for c in r] for r in rows]
    if md:
        out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
        out += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(out)
    w = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    line = lambda r: "  ".join(c.ljust(w[i]) for i, c in enumerate(r))  # noqa: E731
    return "\n".join([line(headers), line(["-" * x for x in w])] + [line(r) for r in rows])


def rng(vals, fmt="{:.1f}") -> str:
    """'nom [opt-cons]' from a 3-list ordered like CASES."""
    lo, hi = min(vals[0], vals[2]), max(vals[0], vals[2])
    return f"{fmt.format(vals[1])} [{fmt.format(lo)}-{fmt.format(hi)}]"


def _h(title: str, md: bool, level: int = 3) -> str:
    return f"\n{'#' * level} {title}\n" if md else f"\n== {title} ==\n"


def render(res: dict, cfg: dict, archs: list[str] | None = None, md: bool = False) -> str:
    A = res["architectures"]
    archs = archs or list(A)
    s = []

    # ---------------------------------------------------------------- requirements
    s.append(_h("Impulse requirements (system mass 100 kg, as simulated)", md))
    rq = res["requirements_at_sim_mass"]
    r = rq["nom"]
    s.append(table(["quantity", "value [N*s]", "source"], [
        ["sim pod use", f"{r['sim_pod']:.0f}", f"{r['statistic']} of {r['sim_pod_task']}"],
        ["sim jet use", f"{r['sim_jet']:.0f}", f"{r['statistic']} of {r['sim_jet_task']}"],
        ["ideal braking I*(H=%g m)" % cfg["mission"]["design_drop_height"],
         f"{r['fp_braking_ideal']:.0f}", "paper eq. (impulse)"],
        ["I* x sim overhead", rng([q["fp_pod"] for q in rq.values()], "{:.0f}"),
         "overhead " + rng([q["fp_overhead"] for q in rq.values()], "{:.2f}")],
        ["design margin", rng([q["margin"] for q in rq.values()], "{:.2f}"), "x"],
        ["**pod requirement**" if md else "POD REQUIREMENT",
         rng([q["pod_impulse"] for q in rq.values()], "{:.0f}"),
         f"basis: {r['pod_basis']}; spec budget {cfg['mission']['spec_budgets']['pod_impulse']:.0f}"],
        ["**jet requirement**" if md else "JET REQUIREMENT",
         rng([q["jet_impulse"] for q in rq.values()], "{:.0f}"),
         f"spec budget {cfg['mission']['spec_budgets']['jet_impulse']:.0f} (binding in T2/T3)"],
    ], md))
    s.append("\nRequirements are re-derived at each architecture's own system mass "
             "(impulse scales with mass), see the mass budgets below.")

    s.append(_h("Ideal braking impulse I*(H) (100 kg, v_t 1.2 m/s at 0.3 m, a = 9 m/s^2)", md))
    s.append(table(["H [m]", "impact v, no thrust [m/s]", "brake starts h_b [m]", "v0 [m/s]",
                    "I* [N*s]", "brake thrust [N]"],
                   [[f"{b['H']:.1f}", f"{b['impact_v_no_thrust']:.2f}", f"{b['h_b']:.2f}",
                     f"{b['v0']:.2f}", f"{b['I']:.0f}", f"{b['thrust_brake']:.0f}"]
                    for b in res["braking_table"]], md))
    j = res["jump"]
    s.append(f"\nT2 jump (thrust only): take-off {j['v_takeoff']:.2f} m/s, burn {j['t_burn']:.2f} s, "
             f"{j['I_thrust_only']:.0f} N*s, minus leg push -> {j['I']:.0f} N*s, plus landing "
             f"braking {j['I_landing']:.0f} N*s = {j['I_total']:.0f} N*s "
             f"(sim: pods 1130 N*s mean, with fans supplying part of the 2.9 kN).")

    # ---------------------------------------------------------------- propellants
    s.append(_h("Propellant options for the nominal requirement "
                f"(pods {r['pod_impulse']:.0f} N*s / 3 kN peak, jets {r['jet_impulse']:.0f} N*s)", md))
    rows = []
    for p in res["propellants"]:
        pod = p["pod"]
        isp = pod["isp"][1]
        jet = p["jet"]
        rows.append([p["label"], f"{isp:.0f}" if isp else "-",
                     rng(pod["propellant"], "{:.2f}"), rng(pod["total"], "{:.2f}"),
                     rng(jet["total"], "{:.2f}") if jet else "-", p["exhaust"], p["response"]])
    s.append(table(["option", "Isp [s]", "pod propellant [kg]", "pod prop+storage [kg]",
                    "jet prop+storage [kg]", "exhaust [C]", "response"], rows, md))
    s.append("\nStorage = tank/casing/pressurant (+ engines for the turbojet, + motors and "
             "power-limited battery for the fans); thruster valves are excluded here.")
    co2 = next(p for p in res["propellants"] if p["key"] == "co2_cold_gas")
    if co2["jet"]["boil_power_W"][1]:
        s.append(f"CO2 at the jets' peak flow must boil ~{co2['jet']['boil_power_W'][1] / 1e3:.0f} kW "
                 "of liquid: the tank chills and the flow collapses, so CO2 suits only low duty.")

    # ---------------------------------------------------------------- fans
    s.append(_h("Electric ducted fans: momentum theory (300 N per fan)", md))
    s.append(table(["duct D [cm]", "P_ideal [kW]", "P_shaft [kW] (FoM 0.70 [0.75-0.60])",
                    "induced v [m/s]", "x spec 1.2 kW"],
                   [[f"{f['diameter'] * 100:.0f}", f"{f['p_ideal'] / 1e3:.1f}",
                     rng([p / 1e3 for p in f["p_shaft"]]), f"{f['v_induced']:.0f}",
                     f"{f['ratio_to_spec']:.0f}x"] for f in res["fan_sweep"]], md))
    if "A" in A:
        fa = A["A"]["cases"]["nom"]["fans"]
        s.append(f"\nP_ideal = sqrt(T^3 / (2 rho A)). 1.2 kW at 300 N would need a "
                 f"{fa['point']['diameter_for_spec_power']:.1f} m rotor. Two fans draw "
                 f"{fa['power_total_elec'] / 1e3:.0f} kW electrical: battery is power-limited "
                 f"({fa['battery_mass_power']:.1f} kg vs {fa['battery_mass_energy']:.2f} kg for "
                 f"150 Wh; 150 Wh would be discharged at {fa['spec_c_rate']:.0f} C).")

    # ---------------------------------------------------------------- nozzles
    jv = res["jet_valve"]
    s.append(_h("Micro-jet nozzle / valve sizing (40 N cold-gas N2)", md))
    s.append(table(["Pc [bar]", "A* [mm^2]", "d* [mm]", "mdot = F/(Isp g0) [g/s]",
                    "mdot = Pc A*/c* [g/s]", "Isp = C_F c*/g0 [s]", "exit T [C]",
                    "peak flow, 25 % of 48 jets [kg/s]"],
                   [[f"{n['pc_bar']:.0f}", f"{n['throat_mm2']:.1f}", f"{n['throat_d_mm']:.2f}",
                     f"{n['mdot_isp'] * 1e3:.0f}", f"{n['mdot_cstar'] * 1e3:.0f}",
                     f"{n['isp_cstar']:.0f}", f"{n['exit_T_C']:.0f}", f"{n['peak_flow_all']:.2f}"]
                    for n in res["nozzles"]], md))
    s.append(f"\nNominal design: Pc {jv['pc_bar']:.0f} bar, throat {jv['throat_d_mm']:.1f} mm, "
             f"{jv['mdot'] * 1e3:.0f} g/s per jet; valve {jv['valve_mass_g']:.0f} g, response "
             f"{jv['response_ms']:.0f} ms, min on-time {jv['min_on_ms']:.0f} ms -> minimum impulse "
             f"bit {jv['min_impulse_bit']:.2f} N*s.  The valve orifice must be larger than the "
             "throat (>= 1.5x area) at that flow, which rules out most miniature 2-3 ms valves.")

    # ---------------------------------------------------------------- architectures
    s.append(_h("Architectures: suit mass budget", md))
    rows = []
    for k in archs:
        c = A[k]["cases"]
        n = c["nom"]
        rows.append([k, rng([c[x]["suit_mass"] for x in CASES]),
                     rng([c[x]["propellant_mass"] for x in CASES], "{:.2f}"),
                     f"{n['pods']['impulse']:.0f}" if n["pods"] else "-",
                     f"{n['jets']['impulse']:.0f}" if n["jets"] else "-",
                     n["n_jets"], f"{n['peak_vertical_thrust']:.0f}", f"{n['thrust_to_weight']:.2f}",
                     A[k]["verdict"]["status"]])
    s.append(table(["arch", "suit mass [kg]", "propellant [kg]", "pod I [N*s]", "jet I [N*s]",
                    "jets", "peak lift [N]", "T/W", "verdict"], rows, md))
    s.append(f"\nWearable-mass target {cfg['mission']['wearable_mass_limit']:.0f} kg; the simulator "
             f"assumes {cfg['mission']['spec_suit_mass']:.0f} kg (suit_default.yaml).")
    for k in archs:
        c = A[k]["cases"]
        n = c["nom"]
        s.append(_h(f"{k}: {n['name']}", md, 4))
        rows = []
        for i, it in enumerate(n["items"]):
            vals = [c[x]["items"][i]["mass"] for x in CASES]
            rows.append([it["subsystem"], it["item"], rng(vals, "{:.2f}"), it["note"]])
        rows.append(["**total**" if md else "TOTAL", "", rng([c[x]["suit_mass"] for x in CASES], "{:.2f}"),
                     f"system {n['system_mass']:.1f} kg"])
        s.append(table(["subsystem", "item", "mass [kg]", "note"], rows, md))
        v = A[k]["verdict"]
        s.append("")
        for b in v["blocking"]:
            s.append(f"- BLOCKING: {b}")
        for b in v["caveats"]:
            s.append(f"- caveat: {b}")
        if n["mvp"]:
            m = n["mvp"]
            s.append(f"- MVP sizing: {m['H']:.1f} m fall to {cfg['architectures']['D']['mvp']['impact_speed']} m/s: "
                     f"fire at {m['fire_height']:.2f} m, {m['burn_time'] * 1e3:.0f} ms at "
                     f"{m['thrust_per_pod']:.0f} N per pod, ideal {m['ideal']:.0f} N*s, "
                     f"with margin {m['impulse']:.0f} N*s")

    s.append(_h("Fall-protection MVP impulse vs height (target impact 2 m/s)", md))
    s.append(table(["H [m]", "impact v, no thrust [m/s]", "fire at [m]", "burn [ms]",
                    "thrust per pod [N]", "ideal I [N*s]", "with margin [N*s]"],
                   [[f"{m['H']:.1f}", f"{m['impact_v_no_thrust']:.2f}", f"{m['fire_height']:.2f}",
                     f"{m['burn_time'] * 1e3:.0f}", f"{m['thrust_per_pod']:.0f}",
                     f"{m['ideal']:.0f}", f"{m['impulse']:.0f}"] for m in res["mvp_table"]], md))

    # ---------------------------------------------------------------- cross-check
    if res.get("sim_crosscheck"):
        s.append(_h("Cross-check against the simulator configs", md))
        rows = [[x["config"], f"{x['suit_mass']:.1f}", f"{x['propellant_mass']:.2f}",
                 f"{x['pod_impulse']:.0f}", f"{x['jet_impulse']:.0f}", x["n_jets"],
                 "yes" if x["fans"] else "no",
                 f"{x['pod_tau'] * 1e3:.0f}" if x["pod_tau"] else "-"] for x in res["sim_crosscheck"]]
        for k in [a for a in ("B", "C") if a in A]:
            c = A[k]["cases"]
            n = c["nom"]
            rows.append([f"sizing arch {k}", rng([c[x]["suit_mass"] for x in CASES]),
                         f"{n['propellant_mass']:.2f}", f"{n['pods']['impulse']:.0f}",
                         f"{n['jets']['impulse']:.0f}", n["n_jets"], "no", "-"])
        s.append(table(["config", "suit mass [kg]", "propellant [kg]", "pod I [N*s]", "jet I [N*s]",
                        "jets", "fans", "pod tau [ms]"], rows, md))

    # ---------------------------------------------------------------- embedded
    al = res["allocator"]
    s.append(_h(f"Allocator on embedded targets ({al['n']} thrusters, 500 Hz)", md))
    meas = (f"measured here {al['x86_measured_ms']:.2f} ms/call (warm-started, Python/NumPy)"
            if al["x86_measured_ms"] else "not measured (run with --bench)")
    s.append(f"x86 reference {al['x86_reference_ms']} ms/call (paper, Python/NumPy); {meas}. "
             f"Budget {al['budget_ms']:.1f} ms of the {al['period_ms']:.0f} ms period. Flop model: "
             f"{rng(al['flops'], '{:.0f}')} flop/call.")
    s.append(table(["platform (compiled C)", "ms/call", "status"],
                   [[p["platform"], rng(p["ms"], "{:.2f}"), p["status"]] for p in al["platforms"]], md))

    # ---------------------------------------------------------------- costs
    for key, title in (("bom_B", "Rough BOM, architecture B (prototype, USD, order of magnitude)"),
                       ("phase1", "Phase 1 first purchases, weeks 1-4 (USD, order of magnitude)"),
                       ("pod_stand", "Lift-pod stand purchases, weeks 5-10 (USD, order of magnitude)")):
        b = res[key]
        s.append(_h(title, md))
        has_week = any(x.get("week") for x in b["rows"])
        hdr = (["week"] if has_week else []) + ["item", "qty", "unit", "total"]
        rows = [([x["week"]] if has_week else []) +
                [x["item"], x["qty"], f"{x['unit'][0]:,.0f}-{x['unit'][1]:,.0f}",
                 f"{x['total'][0]:,.0f}-{x['total'][1]:,.0f}"] for x in b["rows"]]
        rows.append(([""] if has_week else []) +
                    ["**total**" if md else "TOTAL", "", "", f"{b['total'][0]:,.0f}-{b['total'][1]:,.0f}"])
        s.append(table(hdr, rows, md))
    return "\n".join(s).strip() + "\n"


def update_doc(path, generated: str) -> bool:
    """Replace the text between the BEGIN/END markers of ``path``; False if markers missing."""
    from pathlib import Path
    p = Path(path)
    text = p.read_text()
    if BEGIN not in text or END not in text:
        return False
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    p.write_text(head + BEGIN + "\n\n" + generated + "\n" + END + tail)
    return True
