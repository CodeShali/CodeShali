# Hardware sizing and build guide

What it takes to physically build the mobility suit, derived from first principles and from the
simulator's measured gas use. All numbers come from `scripts/size_suit.py` and the assumptions in
`configs/sizing.yaml`. Every assumption there is a range `[optimistic, nominal, conservative]`, so
masses are quoted as **nominal [optimistic–conservative]**. Re-generate the tables at the end of this
file after changing any assumption:

```bash
python scripts/size_suit.py                 # text tables (all architectures)
python scripts/size_suit.py --arch B        # one architecture
python scripts/size_suit.py --json runs/sizing.json
python scripts/size_suit.py --update-doc    # rewrite the generated section below (also times the allocator)
```

> The component masses, Isp values and costs are **typical textbook and catalogue figures, not
> measurements of specific parts**. Use them to make decisions at the order-of-magnitude level.
> Replace them with datasheet values, then with bench measurements (Phase 1), as you go.

---

## 1. Bottom line

| arch | what | suit mass [kg] | propellant [kg] | verdict |
|---|---|---|---|---|
| **A** | spec as written: solid pulse pods + 40 cold-gas jets + 2 electric fans | 41 [22–93] | 2.2 | **INFEASIBLE.** Each 300 N fan needs ~27 kW of shaft power [20–42 kW], not 1.2 kW, so the fan system alone weighs ~26 kg |
| **B** | **recommended baseline:** solid pulse pods + 48 cold-gas N2 jets (incl. 4 side jets), no fans | **16.0 [9.0–27.6]** | 2.2 | feasible with caveats (solid pods only fire in pulses, and the suit weighs 16 kg, not 9) |
| **C** | throttleable alternative: 90 % H2O2 pods + 48 cold-gas N2 jets | 17.2 [9.9–29.4] | 2.8 | feasible with caveats (pod response 10–50 ms; peroxide is a restricted oxidizer) |
| D | optional fallback, fall-protection MVP: airbags + 2 small solid pods, 1–2 m falls | 3.7 [2.3–6.2] | 0.3 | feasible, with no attitude control |

Key findings:

* **The fans are physically impossible as specified.** Actuator-disk momentum theory gives
  P_ideal = sqrt(T³/(2ρA)) = 18.7 kW for 300 N through a 20 cm duct. With a figure of merit of 0.7
  that becomes 26.8 kW of shaft power, 22× the spec's 1.2 kW. To make 300 N from 1.2 kW you would
  need a 4.5 m rotor. Two fans draw ~61 kW electrical, which would discharge the 150 Wh battery at
  ~400 C. The battery is therefore sized by power, not energy (~17 kg). The fans contribute almost
  nothing in the simulation (< 0.3 Wh used), so dropping them costs nothing.
  `configs/suit_realistic.yaml` already removes them.
* **A realistic suit weighs ~15–17 kg, not 9 kg.** In B the jet subsystem dominates at ~7 kg:
  48 valves at ~55 g each is 2.6 kg, plus a 4 L 300 bar tank, plumbing and drivers. The other
  agent's `suit_realistic.yaml` (15.1 kg with 40 jets) agrees with B (16.0 kg with 48 jets and a
  larger jet tank).
* **Impulse requirements.** The pod requirement is ~2.1 kN·s at B's mass. That is the
  first-principles 5 m braking impulse (1394 N·s at 100 kg), × the simulated overhead of 1.15,
  × a 1.25 margin. The spec's 2500 N·s is therefore adequate, with some to spare. The jets need
  ~750 N·s: T2's p95 is 560 N·s, × 1.25 margin, × mass scaling. **This is more than the spec's
  600 N·s**, which T2 and the push stress tests already run dry. Size the jet tank at ~4 L at
  300 bar.
* **The pods are cheap in mass.** Pod propellant for 2 kN·s is ~1.1 kg of solid or ~1.6 kg of
  H2O2. The real question is controllability. Solid charges cannot be throttled or stopped once
  lit. Peroxide can be throttled but responds in tens of ms, where the simulator assumes 5 ms.
* **Micro-jet hardware.** One 40 N jet flows ~60 g/s of N2. At a 15 bar plenum it needs a 4.8 mm
  throat (17.8 mm²), so **the valve orifice must be ≥ 6 mm**. That rules out most miniature
  2–3 ms solenoids. If 12 of the 48 jets fire at once, the regulator must pass ~0.7 kg/s. The
  exhaust leaves the nozzle at roughly −130 °C.
* **Embedded compute.** The allocator takes 0.52 ms per call in the paper. Measured in-loop here
  (warm-started, Python), it takes ~0.18 ms. A dense flop model puts a compiled version at ~1 ms
  per call on a Cortex-M7 in float32, which is *tight* against a 0.6 ms budget (30 % of the 2 ms
  period), and at ~0.2 ms on a Jetson-class Cortex-A. Recommendation: run the estimator,
  allocator and controller on a Cortex-A or Jetson-class computer, and use a Teensy 4.1 or STM32H7
  as the hard-real-time I/O co-processor (1 kHz valve sigma-delta modulation, igniters, IMU
  sampling).

## 2. Method (what the calculator does)

1. **Requirements** (`suitsim/sizing/model.py: requirements`). The calculator takes the per-task
   pod and jet impulse statistic (default p95) from `results/m7_*_summary.json`, using only the
   nominal missions (no stress tests). Note that the *max* jet use is clipped at the 600 N·s tank,
   so it is only a lower bound. It then takes the paper's ideal braking impulse
   I*(H) = m(v0−vt)(1+g/a) + m·g·ht/vt at H = 5 m, multiplied by the simulator's measured overhead
   (×1.07 mean, ×1.26 p95). The larger of the two, × the margin, is the requirement. The
   requirement scales linearly with system mass. Because a heavier suit needs more impulse, which
   adds more mass, the suit mass is iterated to a fixed point.
2. **Propellant.** m = I/(Isp·g0). For air-breathing engines, Isp = 1/(TSFC·g0).
   *Correction to the brief:* small model turbojets burn 0.12–0.18 kg/(N·h). The 0.04–0.05 figure
   applies to large turbofans.
3. **Storage.**
   * Cold gas: ρ = P/(Z·R·T), with Z ≈ 1.1 for N2 at 300 bar as a stated assumption. No real-gas
     equation of state is used.
   * Tank mass: m = SF·P·V/(PV/W·g0). The performance factor PV/W = P_burst·V/weight has units of
     length; Type-IV tanks reach 25–40 km, i.e. P_burst·V/m ≈ 2.5–3.9×10⁵ J/kg. Sanity check: a
     6.8 L, 300 bar SCBA cylinder with SF 3 and 25 km comes out at 2.5 kg, which is realistic.
   * Peroxide: a pressure-fed tank plus an N2 pressurant bottle, sized for a regulated feed.
   * Solid: inert mass from the propellant mass fraction (0.6–0.8).
4. **Nozzles.** A* = F/(C_F·Pc), with C_F 1.4–1.6. The calculator cross-checks the mass flow,
   F/(Isp·g0) against Pc·A*/c*, and the Isp, C_F·c*/g0.
5. **Fans.** Momentum theory and figure of merit as above. The battery mass is
   max(E/e_spec, P/p_spec).
6. **Mass budget.** Each architecture's item list is summed. Every item carries its own
   [opt, nom, cons] value.

## 3. Cross-check against the simulator

| simulator assumption (`suit_default.yaml`) | hardware reality | action |
|---|---|---|
| suit 9 kg | 16 kg (B), 17 kg (C) | run T1–T3 with `suit_realistic*.yaml` (15 kg) or heavier |
| fans 300 N at 1.2 kW | 20–42 kW each | remove the fans (done in `suit_realistic.yaml`) |
| pods continuous 0–1500 N, τ = 5 ms | solid: pulses, ~10 ms ignition delay, cannot stop; H2O2: τ ≈ 30 ms | pulse scheduler / τ = 30 ms (done in the realistic configs) |
| jets continuous 0–40 N, τ = 5 ms | on/off valves with 3 ms open and close delays and a 2–10 ms minimum on-time (minimum impulse bit ~0.2 N·s) | sigma-delta valve model (done in the realistic configs) |
| jet tank 600 N·s | T2 p95 560 N·s, with stress tests emptying it | size the jet tank at ~750 N·s (B) |
| 40 jets | the M7 report recommends 48 (+4 side jets) | combine `suit_realistic` with the side-jet thrusters |
| exhaust ignored | pod plumes at 600–2500 °C point down past the legs | plume keep-out and heat shielding in the pod mount design |

## 4. Safety, noise and regulation (qualitative)

* **Exhaust temperature.**
  * Cold gas exits *below* ambient, around −100 to −180 °C depending on Pc. Expect frost and ice
    on nozzles and skin contact injuries, and use dry N2 (dew point < −60 °C).
  * 90 % H2O2 decomposes to steam and O2 at ~600–750 °C: a scald hazard, and an oxygen-enriched
    fire hazard.
  * Solid gas generators run at 1500–2500 °C in the chamber and shed particulates (HCl if the
    propellant is AP-based).
  * **The pods fire downward past the wearer's legs.** Plume geometry and heat shielding are a
    primary safety design item.
* **Noise** (rough figures). A 40 N choked cold-gas jet is ~110–125 dB at 1 m. Solid and peroxide
  pods are ~130–150 dB peak. Micro-turbojets run at 120–140 dB continuously. Hearing protection
  is mandatory for every test, and the wearer needs it too.
* **Pressure vessels.** Use only certified composite cylinders (e.g. DOT/TC special permits,
  ISO 11119-3, EN 12245 / TPED in the EU). Respect the hydrostatic re-test interval (typically
  5 years) and the service life (15 years or NLL). Never modify a cylinder. Fill only through a
  proper fill station. Any system you assemble that is above the pressure-volume thresholds may
  fall under pressure-equipment rules (e.g. the EU PED 2014/68/EU). Fit a relief valve downstream
  of every regulator and a burst disc on every tank.
* **Pyrotechnics.** Solid charges and airbag inflators are explosives or pyrotechnic articles.
  Making or buying custom gas generators usually needs an explosives licence or permit. Examples:
  US ATF and state fire marshal; EU 2013/29/EU categories plus national authority approval; UK
  Explosives Regulations 2014. Commercially certified hobby motors are the easiest legal
  surrogate for bench work. *This is general orientation, not legal advice: check with your local
  authority before purchase.*
* **90 % H2O2.**
  * It is a strong oxidizer (DOT 5.1/8). In the EU, Regulation 2019/1148 restricts > 12 % to
    professional users, and in the US concentrations ≥ 35 % are a DHS chemical of interest.
  * Few suppliers sell 90 % to individuals.
  * It needs passivated aluminium, 316 SS, PTFE or PE wetted parts, a water deluge, and no
    organics near the stand.
* **Human testing** needs an ethics/IRB review, medical oversight, insurance, and a rated
  fall-arrest system (Phase 4).

---

## 5. Build guide: the full suit (architecture B baseline, C as the throttleable alternative)

The team is two people. **P1** covers mechanical work, the stands and range safety. **P2**
covers electronics, DAQ, firmware and the simulator. Phases overlap, and the pod paperwork starts
in week 1 because licensing has the longest lead time.

### Phase 0: make the simulator hardware-realistic (ongoing, P2, ~1 week up front)

* Run T1, T2 and T3 with `configs/task_*_realistic.yaml` (solid pulses) and
  `task_*_realistic_peroxide.yaml` (H2O2). Compare the success rate and gas use against the M7
  report.
* Add the 4 side jets to the realistic suit and raise the jet budget to ~750 N·s (arch B).
  Check whether T2 and T3 still empty the tank.
* Feed this calculator's B mass (16 kg nominal; 28 kg conservative) into the realistic configs as
  a sensitivity run.
* Decide B (solid) vs C (peroxide) **by end of week 2**, using the sim results and the licensing
  answers. Both bench paths are planned below.

### Phase 1: bench tests (weeks 1–10)

| week | P1 (mechanical / safety) | P2 (electronics / software) | exit criterion |
|---|---|---|---|
| 1 | Hazard analysis (FMEA) and written test procedures. Order the week-1 list. Build the single-jet stand (extrusion frame, S-beam load cell, polycarbonate shield). **Send licensing enquiries** for solid charges and for H2O2. | DAQ at ≥ 1 kHz/ch, 10 kHz preferred. Log valve command, load cell, plenum and tank pressure, and gas temperature on one clock. Logging script and log template (below). | Stand calibrated with dead weights, 5 points, < 1 % nonlinearity. Tap test: stand natural frequency ≥ 300 Hz (it must resolve a 1–5 ms rise). |
| 2 | **(a) Single jet.** Steady thrust vs Pc (5/10/15/20 bar) × 3 throat sizes. Isp from tank mass loss. | Teensy 4.1 valve driver (MOSFET, flyback, current sense). Step-response runs: open delay, rise, close delay, tail-off. Min-on-time sweep (1–20 ms pulses, 100 reps each). | Thrust within 40 N ± 15 % at the design Pc; delays ≤ 4.5 ms; min on-time ≤ 5 ms; Isp ≥ 63 s (criteria table below). **Select the valve and nozzle.** |
| 3 | Blow-down run: 300 bar tank → regulator → jet at a T2-like duty (~300 N·s). Log Isp vs tank temperature. Valve endurance, 10⁵ cycles. Build the **8-jet attitude module** (manifold, 6 mm lines, 8 valves on a rigid plate). | 1 kHz sigma-delta modulation identical to the simulator's valve model: check that the mean thrust is linear in duty. Port the allocator to the module's 8 columns. | Mean thrust error vs duty < 5 %. Isp drop over a blow-down ≤ 10 %. No valve failures in 10⁵ cycles. |
| 4 | **(b) 8-jet module** on a 6-axis F/T sensor. Measure each jet's force and moment (one column of the allocation matrix A). Measure plenum droop with all 8 firing (0.48 kg/s). | Wrench-tracking test: command random feasible wrenches through the allocator and compare measured vs commanded. | Measured A columns within 10 % / 1 cm of the CAD values. Plenum droop < 10 % at full flow. Wrench tracking error < 10 %. |
| 5–6 | Mount the module on a 1-DoF, then 3-DoF, gimbal with torso-equivalent inertia (2–5 kg·m²). **Start the pod stand build**: steel frame, 5 kN cell, ground anchors, barricade. | Closed-loop attitude control on the gimbal. The estimator and allocator run on the target (start of Phase 2). Compare the step response with the simulator, driven by the measured valve model. | Attitude step response within 20 % of the sim prediction (overshoot, settling). No deadline misses at 500 Hz. |
| 7–10 | **(c) Single lift pod** on the heavy stand (see the safety setup). *Solid path:* certified commercial G-class motors as surrogates (80–160 N·s, close to the 156 N·s charge) to qualify the stand and firing system, then licensed custom charges. *Peroxide path:* cold-flow water tests, then a vendor catalyst-bed thruster. | Remote firing box and interlock firmware. Chamber-pressure and thrust DAQ at ≥ 10 kHz. Analysis: impulse, burn time, ignition delay, Isp, repeatability over ≥ 10 firings. | Charge impulse 156 N·s ± 5 %, ignition delay ≤ 15 ms, mean thrust 1500 N ± 15 % (solid). τ ≤ 45 ms, Isp ≥ 117 s (peroxide). |

**Safety setup for the cold-gas stands (a, b)**

* Relief valve downstream of the regulator, set at ~1.1× the regulated pressure. Burst disc on
  the tank.
* Lines and fittings rated ≥ 4× working pressure. Whip checks on hoses.
* A bleed valve that can be reached from behind the shield. Nobody stands in line with a nozzle.
* 10 mm polycarbonate shield, hearing protection and face shields.
* Dry N2 only. Depressurise before touching any fitting.

**Safety setup for the lift-pod stand (c)**

* **Remote firing.** Use a two-step arm: a range-safety key switch, held by the safety officer,
  plus a firing button. Test continuity at a current below the igniter's no-fire current. Fire
  from ≥ 15 m by cable, or by RF with a hardware interlock. Run a spoken countdown, keep a
  logbook, and allow only two people on the range.
* **Standoff distances.** Amateur-rocketry codes (NAR/Tripoli, NFPA 1122/1127) set roughly 9 m
  for G-class and 15 m for H/I-class motors. Treat those as the *floor*. Use ≥ 15–30 m behind a
  sandbag or concrete barricade for custom charges, and keep the exclusion zone clear downrange
  and behind the stand.
* **Blast shield and barricade** between the stand and the operators. The stand is anchored to
  the ground or to a concrete block rated well above 1.5 kN plus the peak transient, with no
  combustibles within 10 m.
* **Misfire procedure.** Disarm and remove the key first. Wait at least 60 s (rocketry-code
  minimum), and longer for larger charges, before approaching.
* **Fire safety.** Extinguishers (ABC + water), plus a water deluge for peroxide, and a burn and
  first-aid kit.
* **Licensing.** Confirm in week 1 what your jurisdiction requires to buy and store motors and
  charges (storage magazine?) and 90 % H2O2, and to fire them at your test space (fire marshal or
  local permit, landowner and insurer approval). See section 4.

**First purchases, weeks 1–4.** The full itemised list is in the generated table
"Phase 1 first purchases" below: roughly **$2.7k–20k**, depending on DAQ and F/T sensor class and
on whether you buy a high-speed camera. The pod-stand list for weeks 5–10 is a further
$4.3k–26.5k. That figure includes *both* the solid and the peroxide path, so you only pay for the
one you choose.

#### Test-log template

One row per run. Keep the raw DAQ file next to it (`bench_logs/<test-id>.csv`, with a header
recording the calibration constants).

| field | example | notes |
|---|---|---|
| test id / date / operators | T-J-014 / 2026-11-03 / P1, P2 | id prefix: J = jet, M = module, P = pod |
| article | valve model, serial; nozzle d* 4.8 mm | |
| conditions | tank 280 bar, Pc set 15 bar, gas T 285 K, ambient 18 °C | |
| command | 50 × 20 ms pulses @ 10 Hz | |
| F_ss [N] (mean ± sd) | 39.2 ± 0.8 | steady part of each pulse |
| open delay / rise τ (63 %) / close delay [ms] | 2.8 / 0.9 / 3.1 | from command edge to thrust |
| min on-time [ms] | 4 | shortest pulse with impulse within ±20 % of F·t and sd ≤ 10 % |
| impulse bit [N·s] | 0.16 | at the min on-time |
| Δm [g] / ∫F dt [N·s] / Isp [s] | 61.2 / 40.1 / 66.8 | Isp = ∫F dt/(Δm·g0); Δm ≥ 50 g for 0.2 % resolution |
| pass / fail, anomalies | pass | link to video |

#### Pass criteria, tied to the simulator's YAML

A measured value inside the simulator's domain-randomisation band means the existing sim results
already cover the hardware. The bands are `randomization.thruster_force_frac: 0.15` and
`response_delay_frac: 0.5` in `configs/task_drop.yaml`. A value outside the band means you
update the YAML and re-run the evaluations.

| measured | YAML key (`configs/suit_realistic.yaml`) | sim value | pass |
|---|---|---|---|
| jet steady thrust | `suit.actuator_classes.micro_jet.f_max` | 40 N | 40 N ± 15 %, unit-to-unit sd ≤ 5 % |
| valve open / close delay | `...micro_jet.valve.open_delay` / `close_delay` | 3 ms / 3 ms | ≤ 4.5 ms (+50 %) |
| chamber rise / tail-off | `...micro_jet.valve.rise_tau` | 1 ms | ≤ 1.5 ms |
| valve min on-time | `...micro_jet.valve.min_on_time` | 5 ms | ≤ 5 ms |
| jet first-order lag (continuous model) | `...micro_jet.tau` | 5 ms | ≤ 7.5 ms |
| jet Isp | `suit.propellant.impulse_gas_jets.propellant_isp` | 70 s | ≥ 63 s (−10 %) |
| charge impulse / burn time | `...lift_pod.solid_pulses.charge_impulse` / `burn_time` | 156 N·s / 0.104 s | ± 5 % / ± 20 % |
| ignition delay | `...lift_pod.solid_pulses.ignition_delay` | 10 ms | ≤ 15 ms |
| pod mean thrust | `...lift_pod.f_max` | 1500 N | ± 15 % |
| pod Isp (solid) | `suit.propellant.impulse_gas_pods.propellant_isp` | 200 s | ≥ 180 s |
| peroxide throttle lag / Isp | `suit_realistic_peroxide.yaml: ...lift_pod.tau` / `propellant_isp` | 30 ms / 130 s | ≤ 45 ms / ≥ 117 s |
| hardware masses | `suit.masses.jet_tank`, `pod_casings`, ... | 4.0, 2.5 kg | weigh the parts |

#### Writing measurements back

1. After each test campaign, put the **measured mean** in the YAML key from the table and add a
   provenance comment, e.g.
   `open_delay: 0.0028   # measured T-J-014..020, 2026-11-03, n=350, sd 0.2 ms`.
   Edit `configs/suit_realistic.yaml` directly. If you want to keep the literature baseline, use
   a child file `configs/suit_measured.yaml` with `base: suit_realistic.yaml` holding only the
   measured keys, and point the `task_*_realistic.yaml` files at it.
2. If the measured **spread** exceeds the randomisation band (e.g. thrust sd > 5 %, so 3 sd >
   15 %), widen `randomization.thruster_force_frac` or `response_delay_frac` in the realistic task
   configs to cover it.
3. Put the measured masses and Isp into `configs/sizing.yaml`, collapsing those ranges to the
   measured value ± its uncertainty. Then run `python scripts/size_suit.py --update-doc`.
4. Re-run the evaluations, e.g.
   `python scripts/evaluate.py --task configs/task_drop_realistic.yaml --episodes 1000 --workers 4 --out runs/eval_t1_measured`.
   The gate for moving to the next phase is T1 ≥ 95 % and T2 and T3 within budget. Commit the
   YAML change and the run summary together.

### Phase 2: hardware-in-the-loop on the embedded target (weeks 5–12, P2)

* Run the estimator, controller and allocator on the target at 500 Hz. MuJoCo runs on a PC and
  streams sensor packets (IMU, altimeter, foot pressure, encoders) over Ethernet or USB at 500 Hz.
  The target returns thruster commands, which the PC applies to the simulated suit.
* Measure end-to-end latency, jitter, deadline misses and float32-vs-float64 allocator agreement,
  using logged wrenches against the Python reference.
* **Expected allocator cost** (generated table below): ~0.2 ms on a Jetson-class Cortex-A, and
  ~1 ms on a Cortex-M7 in float32, which is tight. Hence the recommended split: a Cortex-A or
  Jetson for estimation and control, and a Teensy 4.1 or STM32H7 for 1 kHz valve modulation,
  igniter sequencing and IMU sampling.
* Then drive the real 8-jet module (Phase 1b) from the HIL loop. This is the first true
  hardware-in-the-loop test.

### Phase 3: instrumented dummy drop rig

* **Dummy.** A 90–100 kg articulated dummy (rescue-training class), ballasted to human segment
  masses, carrying the suit IMUs plus pelvis and head accelerometers.
* **Rig.** A gantry or crane with a quick release and a **catch line** (dynamic rope + energy
  absorber). Set the slack so the line arrests the dummy ~0.3 m above the ground if the suit
  fails. Crash mats underneath. The pod safety rules (standoff, remote arm) apply.
* **Heights** 0.5 → 1.0 → 1.5 → 2.0 m. Start upright, then add 10°, 20° and 30° tilt, then spin.
  Advance only after N consecutive passes at each step.
* **Pass criteria** are the simulator's limits: vertical impact ≤ 2 m/s, horizontal ≤ 1 m/s,
  tilt ≤ 20°, feet first.
* **Use of the rig.** Compare the measured trajectories and impulse with the simulator's
  prediction for the same initial state. The rig *validates the simulator*. It cannot prove
  reliability: 10/10 passes has a Wilson lower bound of only ~72 %. The statistics come from the
  validated simulator.

### Phase 4: tethered human

* Get ethics/IRB approval, a medical observer and insurance first.
* Use an overhead tether with a rated fall-arrest harness, and enforce exhaust keep-out zones.
* **Order of tests:**
  1. Standing push recovery using the jets only (lowest energy, cold gas).
  2. Short drops (0.3–0.5 m onto mats) with the jets.
  3. Pods, only after an extensive dummy record at the same condition.

### Optional fallback: fall-protection MVP (architecture D)

If the full suit stalls, a 3.7 kg [2.3–6.2] kit can still brake a 2 m fall to 2 m/s. It consists
of airbags plus two small solid pulse pods, ~1.4 kN each for ~160 ms, fired at ~0.6 m height.
That takes ~460 N·s ideal, ~575 N·s with margin. It has no attitude control, so it only works for
near-upright falls. The airbags cover the rest. Phase 1(c) and Phase 3 test this hardware
directly.

---

## 6. Generated tables

<!-- BEGIN GENERATED: scripts/size_suit.py -->

### Impulse requirements (system mass 100 kg, as simulated)

| quantity | value [N*s] | source |
|---|---|---|
| sim pod use | 1515 | p95 of m7_t1_baseline |
| sim jet use | 560 | p95 of m7_t2_jump |
| ideal braking I*(H=5 m) | 1394 | paper eq. (impulse) |
| I* x sim overhead | 1604 [1492-1757] | overhead 1.15 [1.07-1.26] |
| design margin | 1.25 [1.15-1.50] | x |
| **pod requirement** | 2005 [1742-2636] | basis: first-principles I*(H); spec budget 2500 |
| **jet requirement** | 701 [645-841] | spec budget 600 (binding in T2/T3) |

Requirements are re-derived at each architecture's own system mass (impulse scales with mass), see the mass budgets below.

### Ideal braking impulse I*(H) (100 kg, v_t 1.2 m/s at 0.3 m, a = 9 m/s^2)

| H [m] | impact v, no thrust [m/s] | brake starts h_b [m] | v0 [m/s] | I* [N*s] | brake thrust [N] |
|---|---|---|---|---|---|
| 0.5 | 3.13 | 0.37 | 1.62 | 333 | 1881 |
| 1.0 | 4.43 | 0.63 | 2.71 | 560 | 1881 |
| 1.5 | 5.42 | 0.89 | 3.47 | 719 | 1881 |
| 2.0 | 6.26 | 1.15 | 4.09 | 849 | 1881 |
| 3.0 | 7.67 | 1.67 | 5.11 | 1062 | 1881 |
| 4.0 | 8.86 | 2.19 | 5.96 | 1239 | 1881 |
| 5.0 | 9.90 | 2.71 | 6.70 | 1394 | 1881 |

T2 jump (thrust only): take-off 7.84 m/s, burn 0.41 s, 1184 N*s, minus leg push -> 959 N*s, plus landing braking 404 N*s = 1364 N*s (sim: pods 1130 N*s mean, with fans supplying part of the 2.9 kN).

### Propellant options for the nominal requirement (pods 2005 N*s / 3 kN peak, jets 701 N*s)

| option | Isp [s] | pod propellant [kg] | pod prop+storage [kg] | jet prop+storage [kg] | exhaust [C] | response |
|---|---|---|---|---|---|---|
| cold-gas N2 (300 bar) | 68 | 3.01 [2.80-3.41] | 6.37 [4.18-9.34] | 2.39 [1.56-3.53] | -100 to -150 at the nozzle exit (below ambient) | solenoid 2-10 ms |
| cold-gas CO2 (liquid, ~57 bar) | 58 | 3.52 [3.35-4.09] | 4.56 [3.83-5.99] | 1.72 [1.40-2.29] | -60 to -80 (dry-ice snow possible) | solenoid 2-10 ms, but flow limited by boiling (tank chills) |
| solid gas generator / small solid motor | 200 | 1.02 [0.93-1.14] | 1.46 [1.16-1.89] | 0.51 [0.41-0.66] | 1500-2500 (chamber), particulates | ignition 5-30 ms; NOT throttleable (pulse grains / pintle) |
| 90 % H2O2 monopropellant | 135 | 1.51 [1.36-1.70] | 2.05 [1.63-2.76] | 0.94 [0.70-1.36] | ~600-750 (superheated steam + O2) | 10-50 ms warm bed, 100+ ms cold start |
| micro turbojet (kerosene) | 2447 | 0.08 [0.07-0.10] | 30.99 [25.87-44.12] | 5.23 [4.41-7.33] | 500-750 (EGT), 120-140 dB | seconds (spool-up) - must already be running |
| electric ducted fans (same peak lift as pods) | - | 0.00 [0.00-0.00] | 130.09 [66.41-345.03] | - | ambient air, 100-115 dB | 50-150 ms spool |

Storage = tank/casing/pressurant (+ engines for the turbojet, + motors and power-limited battery for the fans); thruster valves are excluded here.
CO2 at the jets' peak flow must boil ~194 kW of liquid: the tank chills and the flow collapses, so CO2 suits only low duty.

### Electric ducted fans: momentum theory (300 N per fan)

| duct D [cm] | P_ideal [kW] | P_shaft [kW] (FoM 0.70 [0.75-0.60]) | induced v [m/s] | x spec 1.2 kW |
|---|---|---|---|---|
| 15 | 25.0 | 35.7 [33.3-41.6] | 83 | 30x |
| 20 | 18.7 | 26.8 [25.0-31.2] | 62 | 22x |
| 25 | 15.0 | 21.4 [20.0-25.0] | 50 | 18x |
| 30 | 12.5 | 17.8 [16.6-20.8] | 42 | 15x |

P_ideal = sqrt(T^3 / (2 rho A)). 1.2 kW at 300 N would need a 4.5 m rotor. Two fans draw 61 kW electrical: battery is power-limited (17.4 kg vs 0.75 kg for 150 Wh; 150 Wh would be discharged at 405 C).

### Micro-jet nozzle / valve sizing (40 N cold-gas N2)

| Pc [bar] | A* [mm^2] | d* [mm] | mdot = F/(Isp g0) [g/s] | mdot = Pc A*/c* [g/s] | Isp = C_F c*/g0 [s] | exit T [C] | peak flow, 25 % of 48 jets [kg/s] |
|---|---|---|---|---|---|---|---|
| 5 | 53.3 | 8.24 | 60 | 63 | 64 | -96 | 0.72 |
| 10 | 26.7 | 5.83 | 60 | 63 | 64 | -128 | 0.72 |
| 20 | 13.3 | 4.12 | 60 | 63 | 64 | -154 | 0.72 |
| 50 | 5.3 | 2.61 | 60 | 63 | 64 | -182 | 0.72 |

Nominal design: Pc 15 bar, throat 4.8 mm, 60 g/s per jet; valve 55 g, response 5 ms, min on-time 5 ms -> minimum impulse bit 0.20 N*s.  The valve orifice must be larger than the throat (>= 1.5x area) at that flow, which rules out most miniature 2-3 ms valves.

### Architectures: suit mass budget

| arch | suit mass [kg] | propellant [kg] | pod I [N*s] | jet I [N*s] | jets | peak lift [N] | T/W | verdict |
|---|---|---|---|---|---|---|---|---|
| A | 41.2 [22.3-93.1] | 2.17 [2.00-2.44] | 2500 | 600 | 40 | 3600 | 2.78 | INFEASIBLE |
| B | 16.0 [9.0-27.6] | 2.22 [1.71-3.47] | 2146 | 750 | 48 | 3000 | 2.86 | FEASIBLE with caveats |
| C | 17.2 [9.9-29.4] | 2.77 [2.10-4.42] | 2169 | 758 | 48 | 3000 | 2.83 | FEASIBLE with caveats |
| D | 3.7 [2.3-6.2] | 0.29 [0.24-0.41] | 575 | - | 0 | 2824 | 3.04 | FEASIBLE with caveats |

Wearable-mass target 20 kg; the simulator assumes 9 kg (suit_default.yaml).

#### A: Spec as written: solid pulse pods + 40 cold-gas jets + 2 electric fans

| subsystem | item | mass [kg] | note |
|---|---|---|---|
| pods | solid propellant | 1.27 [1.16-1.42] | Isp 200 s |
| pods | motor case / nozzle / igniter (inert) | 0.55 [0.29-0.94] | propellant mass fraction 0.70 |
| pods | pod thruster hardware x2 | 0.50 [0.20-1.00] |  |
| jets | N2 gas load | 1.00 [0.88-1.20] | 0.90 kg usable, Isp 68 s |
| jets | Type-IV tank shell | 0.83 [0.32-1.48] | 3.26 L @ 300 bar |
| jets | tank valve / burst disc / fittings | 0.25 [0.15-0.40] |  |
| jets | solenoid valve + nozzle x40 | 2.20 [1.20-3.20] | 55 g each |
| jets | valve drivers + wiring x40 | 0.40 [0.20-0.80] |  |
| jets | feed tubing + fittings | 0.68 [0.29-1.68] |  |
| jets | manifolds | 0.20 [0.10-0.40] |  |
| jets | regulator | 0.30 [0.15-0.60] |  |
| fans | motor + ESC x2 | 7.64 [4.00-16.65] | 26.8 kW shaft each |
| fans | rotor + duct x2 | 1.00 [0.60-1.60] | 20 cm duct |
| fans | fan battery | 17.37 [8.69-50.76] | power-limited |
| structure/other | airbags | 1.50 [1.00-2.50] |  |
| structure/other | compute_sensors | 1.00 [0.60-1.50] |  |
| structure/other | frame | 2.50 [1.50-4.00] |  |
| structure/other | shell | 2.00 [1.00-3.00] |  |
| **total** |  | 41.20 [22.32-93.12] | system 132.2 kg |

- BLOCKING: fans: 27 kW shaft per 300 N fan (range 20-42 kW) vs 1.2 kW in the spec; fan system 26.0 kg; 1.2 kW would need a 4.5 m rotor
- BLOCKING: suit mass 22.3 kg even optimistically (> 20 kg wearable limit)
- caveat: suit is 41.2 kg, not the simulated 9 kg: re-run the sim at this mass
- caveat: solid pods cannot throttle: the sim's continuous 0-1500 N, 5 ms pods must become discrete pulses (controller rework)

#### B: Recommended: solid pulse pods + 48 cold-gas N2 jets (incl. 4 side jets), no fans

| subsystem | item | mass [kg] | note |
|---|---|---|---|
| pods | solid propellant | 1.09 [0.81-1.77] | Isp 200 s |
| pods | motor case / nozzle / igniter (inert) | 0.47 [0.20-1.18] | propellant mass fraction 0.70 |
| pods | pod thruster hardware x2 | 0.50 [0.20-1.00] |  |
| jets | N2 gas load | 1.25 [0.95-1.99] | 1.12 kg usable, Isp 68 s |
| jets | Type-IV tank shell | 1.04 [0.35-2.45] | 4.08 L @ 300 bar |
| jets | tank valve / burst disc / fittings | 0.25 [0.15-0.40] |  |
| jets | solenoid valve + nozzle x48 | 2.64 [1.44-3.84] | 55 g each |
| jets | valve drivers + wiring x48 | 0.48 [0.24-0.96] |  |
| jets | feed tubing + fittings | 0.82 [0.35-2.02] |  |
| jets | manifolds | 0.20 [0.10-0.40] |  |
| jets | regulator | 0.30 [0.15-0.60] |  |
| structure/other | airbags | 1.50 [1.00-2.50] |  |
| structure/other | compute_sensors | 1.00 [0.60-1.50] |  |
| structure/other | frame | 2.50 [1.50-4.00] |  |
| structure/other | shell | 2.00 [1.00-3.00] |  |
| **total** |  | 16.04 [9.03-27.62] | system 107.0 kg |

- caveat: suit is 16.0 kg, not the simulated 9 kg: re-run the sim at this mass
- caveat: solid pods cannot throttle: the sim's continuous 0-1500 N, 5 ms pods must become discrete pulses (controller rework)

#### C: Throttleable: 90 % H2O2 pods + 48 cold-gas N2 jets, no fans

| subsystem | item | mass [kg] | note |
|---|---|---|---|
| pods | 90 % H2O2 | 1.64 [1.20-2.70] | Isp 135 s, 1.18 L |
| pods | peroxide tank | 0.09 [0.02-0.43] | 1.30 L @ 35 bar (Al/lined) |
| pods | N2 pressurant gas | 0.06 [0.03-0.14] | 0.19 L @ 300 bar |
| pods | pressurant bottle | 0.05 [0.01-0.17] |  |
| pods | regulator / vent / check / burst disc | 0.35 [0.20-0.60] |  |
| pods | pod thruster hardware x2 | 1.00 [0.60-1.60] |  |
| jets | N2 gas load | 1.26 [0.96-2.02] | 1.14 kg usable, Isp 68 s |
| jets | Type-IV tank shell | 1.05 [0.35-2.49] | 4.12 L @ 300 bar |
| jets | tank valve / burst disc / fittings | 0.25 [0.15-0.40] |  |
| jets | solenoid valve + nozzle x48 | 2.64 [1.44-3.84] | 55 g each |
| jets | valve drivers + wiring x48 | 0.48 [0.24-0.96] |  |
| jets | feed tubing + fittings | 0.82 [0.35-2.02] |  |
| jets | manifolds | 0.20 [0.10-0.40] |  |
| jets | regulator | 0.30 [0.15-0.60] |  |
| structure/other | airbags | 1.50 [1.00-2.50] |  |
| structure/other | compute_sensors | 1.00 [0.60-1.50] |  |
| structure/other | frame | 2.50 [1.50-4.00] |  |
| structure/other | shell | 2.00 [1.00-3.00] |  |
| **total** |  | 17.19 [9.89-29.35] | system 108.2 kg |

- caveat: suit is 17.2 kg, not the simulated 9 kg: re-run the sim at this mass
- caveat: H2O2 pods: 10-50 ms response (sim assumes 5 ms); 90 % peroxide is a restricted, hazardous oxidizer

#### D: Fall-protection MVP: airbags + 2 small solid pulse pods (1-2 m falls)

| subsystem | item | mass [kg] | note |
|---|---|---|---|
| pods | solid propellant | 0.29 [0.24-0.41] | Isp 200 s |
| pods | motor case / nozzle / igniter (inert) | 0.13 [0.06-0.28] | propellant mass fraction 0.70 |
| pods | pod thruster hardware x2 | 0.50 [0.20-1.00] |  |
| structure/other | airbags | 1.50 [1.00-2.50] |  |
| structure/other | compute_sensors | 0.50 [0.30-0.80] |  |
| structure/other | harness | 0.80 [0.50-1.20] |  |
| **total** |  | 3.72 [2.30-6.19] | system 94.7 kg |

- caveat: solid pods cannot throttle: the sim's continuous 0-1500 N, 5 ms pods must become discrete pulses (controller rework)
- caveat: no attitude control: relies on near-upright falls; airbags cover the rest
- MVP sizing: 2.0 m fall to 2.0 m/s: fire at 0.59 m, 163 ms at 1412 N per pod, ideal 460 N*s, with margin 575 N*s

### Fall-protection MVP impulse vs height (target impact 2 m/s)

| H [m] | impact v, no thrust [m/s] | fire at [m] | burn [ms] | thrust per pod [N] | ideal I [N*s] | with margin [N*s] |
|---|---|---|---|---|---|---|
| 1.0 | 4.43 | 0.26 | 90 | 1412 | 255 | 319 |
| 1.5 | 5.42 | 0.43 | 129 | 1412 | 366 | 457 |
| 2.0 | 6.26 | 0.59 | 163 | 1412 | 460 | 575 |

### Cross-check against the simulator configs

| config | suit mass [kg] | propellant [kg] | pod I [N*s] | jet I [N*s] | jets | fans | pod tau [ms] |
|---|---|---|---|---|---|---|---|
| suit_default.yaml | 9.0 | 0.00 | 2500 | 600 | 40 | yes | 5 |
| suit_realistic.yaml | 15.1 | 2.15 | 2496 | 600 | 40 | no | 5 |
| suit_realistic_peroxide.yaml | 15.3 | 2.84 | 2500 | 600 | 40 | no | 30 |
| sizing arch B | 16.0 [9.0-27.6] | 2.22 | 2146 | 750 | 48 | no | - |
| sizing arch C | 17.2 [9.9-29.4] | 2.77 | 2169 | 758 | 48 | no | - |

### Allocator on embedded targets (48 thrusters, 500 Hz)

x86 reference 0.52 ms/call (paper, Python/NumPy); measured here 0.18 ms/call (warm-started, Python/NumPy). Budget 0.6 ms of the 2 ms period. Flop model: 396288 [165888-1179648] flop/call.
| platform (compiled C) | ms/call | status |
|---|---|---|
| Teensy 4.1 (Cortex-M7 600 MHz), float64 | 3.30 [0.83-19.66] | too slow |
| Teensy 4.1 (Cortex-M7 600 MHz), float32 | 0.99 [0.28-5.90] | tight |
| STM32H7 (Cortex-M7 480 MHz), float32 | 1.32 [0.35-7.86] | tight |
| Jetson Orin Nano (Cortex-A78AE), C/float64 | 0.20 [0.04-1.18] | OK nominal |

### Rough BOM, architecture B (prototype, USD, order of magnitude)

| item | qty | unit | total |
|---|---|---|---|
| fast solenoid valves (2-5 ms, >=6 mm orifice @ 15-20 bar) | 48 | 40-300 | 1,920-14,400 |
| machined/printed-metal micro nozzles | 48 | 10-80 | 480-3,840 |
| Type-IV 300 bar cylinder (~3-5 L, SCBA/paintball class) | 1 | 300-1,200 | 300-1,200 |
| high-flow 300 bar regulator + relief + burst disc | 1 | 150-800 | 150-800 |
| tubing, fittings, manifolds | 1 | 150-600 | 150-600 |
| valve driver boards (MOSFET, flyback, current sense) | 48 | 3-20 | 144-960 |
| solid pulse pods (custom gas generators, dev + certification) | 2 | 2,000-25,000 | 4,000-50,000 |
| airbags + inflators (motorcycle-vest class) | 1 | 400-1,500 | 400-1,500 |
| IMUs (15) | 15 | 10-60 | 150-900 |
| radar altimeter | 1 | 100-600 | 100-600 |
| flight computer (MCU + Jetson-class) | 1 | 80-700 | 80-700 |
| frame, harness, shell (prototype) | 1 | 500-5,000 | 500-5,000 |
| **total** |  |  | 8,374-80,500 |

### Phase 1 first purchases, weeks 1-4 (USD, order of magnitude)

| week | item | qty | unit | total |
|---|---|---|---|---|
| 1 | load cell 100-200 N S-beam + 24-bit/high-rate amplifier (single-jet stand) | 1 | 80-400 | 80-400 |
| 1 | DAQ >= 1 kHz/ch (10 kHz preferred), 16-bit, >= 8 ch (LabJack T7 / NI USB-6000 class) | 1 | 300-1,500 | 300-1,500 |
| 1 | pressure transducers: 0-400 bar (tank) + 0-25 bar (plenum), >= 1 kHz | 2 | 60-400 | 120-800 |
| 1 | type-K thermocouples + amplifier (tank / plenum gas temperature) | 2 | 15-60 | 30-120 |
| 1 | candidate fast solenoid valves (2-3 models, 2-5 ms, >= 4-6 mm orifice) | 6 | 40-300 | 240-1,800 |
| 1 | test nozzles, 3 throat sizes (3, 4, 5.5 mm) | 6 | 10-80 | 60-480 |
| 1 | 300 bar Type-IV cylinder (3-5 L) + fill adapter | 1 | 300-1,200 | 300-1,200 |
| 1 | 300 bar -> 5-25 bar high-flow regulator + relief valve + burst disc + bleed valve | 1 | 150-800 | 150-800 |
| 1 | N2 supply: cylinder rental / fills (or scuba/HPA compressor access) | 1 | 100-600 | 100-600 |
| 1 | stand frame (aluminium extrusion) + 10 mm polycarbonate shield | 1 | 200-1,000 | 200-1,000 |
| 1 | PPE: hearing protection, face shields, cryo gloves, safety glasses | 1 | 50-200 | 50-200 |
| 2 | Teensy 4.1 + MOSFET valve-driver board (8 ch, flyback, current sense) | 1 | 60-250 | 60-250 |
| 2 | precision scale 0.1 g / 10 kg (tank mass loss -> Isp) | 1 | 100-600 | 100-600 |
| 3 | 8 more valves + nozzles of the selected model (8-jet module) | 8 | 50-380 | 400-3,040 |
| 3 | manifold block + 6 mm feed lines + fittings (8-jet module) | 1 | 100-400 | 100-400 |
| 4 | 6-axis force/torque sensor (or 3-4 load-cell plate) for the module wrench map | 1 | 400-4,000 | 400-4,000 |
| 4 | high-speed camera >= 1000 fps (optional; phone slow-mo as fallback) | 1 | 0-3,000 | 0-3,000 |
|  | **total** |  |  | 2,690-20,190 |

### Lift-pod stand purchases, weeks 5-10 (USD, order of magnitude)

| item | qty | unit | total |
|---|---|---|---|
| 5 kN load cell + amplifier (pod stand) | 1 | 150-900 | 150-900 |
| steel pod stand, ground anchors, sandbag/concrete barricade | 1 | 500-2,500 | 500-2,500 |
| remote firing system: key-switch arm, continuity check, >= 15 m cable / RF with interlock | 1 | 150-800 | 150-800 |
| chamber pressure transducer 0-200 bar, >= 10 kHz | 1 | 150-600 | 150-600 |
| certified commercial G-H class motors / gas generators as surrogate charges | 10 | 20-120 | 200-1,200 |
| OR peroxide path: catalyst-bed 1.5 kN thruster + peroxide-grade valve + passivated tank | 1 | 3,000-20,000 | 3,000-20,000 |
| fire extinguishers (ABC + water), deluge water supply, first-aid / burn kit | 1 | 150-500 | 150-500 |
| **total** |  |  | 4,300-26,500 |

<!-- END GENERATED: scripts/size_suit.py -->
