"""First-principles formulas used by the hardware sizing calculator.

Every function is a plain closed-form relation in SI units (N, kg, m, s, Pa, K, W) with no
hidden constants: the assumed *values* live in ``configs/sizing.yaml``.  Keep this module
dependency-free (math only) so the formulas can be checked by hand and unit-tested.
"""
from __future__ import annotations

import math

G0 = 9.80665          # standard gravity [m/s^2] (used for Isp <-> exhaust velocity)


# ----------------------------------------------------------------------------- propellant
def propellant_mass(impulse: float, isp: float, g0: float = G0) -> float:
    """Ideal rocket equation for a small Delta-m: m = I_total / (Isp * g0)  [kg]."""
    if isp <= 0:
        raise ValueError("Isp must be positive")
    return impulse / (isp * g0)


def mass_flow(thrust: float, isp: float, g0: float = G0) -> float:
    """Propellant mass flow at a given thrust: mdot = F / (Isp * g0)  [kg/s]."""
    return thrust / (isp * g0)


def tsfc_to_isp(tsfc_kg_per_N_h: float, g0: float = G0) -> float:
    """Air-breathing engines: thrust-specific fuel consumption [kg/(N*h)] -> fuel Isp [s]."""
    tsfc_si = tsfc_kg_per_N_h / 3600.0          # kg/(N*s)
    return 1.0 / (tsfc_si * g0)


# ----------------------------------------------------------------------------- fans
def disk_area(diameter: float) -> float:
    return math.pi * diameter ** 2 / 4.0


def ideal_induced_power(thrust: float, area: float, rho: float) -> float:
    """Actuator-disk momentum theory (static, open disk): P_ideal = sqrt(T^3 / (2 rho A)) [W]."""
    return math.sqrt(thrust ** 3 / (2.0 * rho * area))


def fan_shaft_power(thrust: float, diameter: float, figure_of_merit: float, rho: float) -> float:
    """Shaft power = ideal induced power / figure of merit (FoM = P_ideal / P_actual)."""
    return ideal_induced_power(thrust, disk_area(diameter), rho) / figure_of_merit


def area_for_power(thrust: float, shaft_power: float, figure_of_merit: float, rho: float) -> float:
    """Inverse of fan_shaft_power: disk area that produces ``thrust`` with ``shaft_power``."""
    p_ideal = shaft_power * figure_of_merit
    return thrust ** 3 / (2.0 * rho * p_ideal ** 2)


# ----------------------------------------------------------------------------- storage
def gas_density(pressure: float, temperature: float, r_specific: float, z: float = 1.0) -> float:
    """Real-gas-ish density rho = P / (Z R T)  [kg/m^3]."""
    return pressure / (z * r_specific * temperature)


def tank_mass(pressure: float, volume: float, performance_factor_m: float,
              safety_factor: float, g0: float = G0) -> float:
    """Composite pressure-vessel shell mass from the performance factor.

    The performance factor is PV/W = P_burst * V / W with W the tank *weight* [N], so it has
    units of length (m); Type-IV carbon/polymer-liner tanks reach ~25-40 km (equivalently
    P_burst*V/m = PV/W * g0 ~ 2.5e5-3.9e5 J/kg).  With P_burst = SF * P_operating:
        m_tank = SF * P * V / (PV/W * g0).
    """
    if performance_factor_m <= 0 or volume < 0:
        raise ValueError("bad tank parameters")
    return safety_factor * pressure * volume / (performance_factor_m * g0)


def blowdown_pressurant_volume(p_tank: float, v_tank: float, p_bottle: float,
                               z_bottle: float = 1.0) -> float:
    """Regulated pressure feed: bottle volume that keeps a liquid tank of volume ``v_tank`` at
    ``p_tank`` until it is empty (isothermal; the bottle ends at ``p_tank``).
        p_b/Z V_b = p_t V_b + p_t V_t  ->  V_b = p_t V_t / (p_b/Z - p_t)."""
    eff = p_bottle / z_bottle - p_tank
    if eff <= 0:
        raise ValueError("bottle pressure must exceed tank pressure")
    return p_tank * v_tank / eff


# ----------------------------------------------------------------------------- nozzles
def throat_area(thrust: float, thrust_coefficient: float, chamber_pressure: float) -> float:
    """A* = F / (C_F * Pc)  [m^2]."""
    return thrust / (thrust_coefficient * chamber_pressure)


def diameter_from_area(area: float) -> float:
    return math.sqrt(4.0 * area / math.pi)


def c_star(gamma: float, r_specific: float, t_chamber: float) -> float:
    """Characteristic velocity of an ideal gas: c* = sqrt(gamma R T) / (gamma * Gamma_term) with
    Gamma_term = sqrt((2/(gamma+1))^((gamma+1)/(gamma-1)))  [m/s]."""
    gterm = math.sqrt((2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (gamma - 1.0)))
    return math.sqrt(gamma * r_specific * t_chamber) / (gamma * gterm)


def isentropic_temperature(t0: float, pressure_ratio: float, gamma: float) -> float:
    """Static temperature after isentropic expansion to p/p0 = pressure_ratio  [K]."""
    return t0 * pressure_ratio ** ((gamma - 1.0) / gamma)


# ----------------------------------------------------------------------------- requirements
def braking_impulse(height: float, mass: float, v_t: float, h_t: float, a: float,
                    g: float = 9.81) -> dict:
    """Ideal braking impulse under the constant-deceleration profile of the paper (eq. impulse):

        v_ref(h) = sqrt(v_t^2 + 2 a (h - h_t)) for h > h_t, v_t below h_t,
        h_b = (2 g H - v_t^2 + 2 a h_t) / (2 (g + a)),  v0 = sqrt(2 g (H - h_b)),
        I*(H) = m (v0 - v_t)(1 + g/a) + m g h_t / v_t.

    ``height`` is the release height of the feet.  Low drops that never reach the profile
    before h_t only pay the constant-speed hold (or nothing if they never reach v_t).
    Returns dict(I, h_b, v0, t_brake, thrust_brake, impact_v_no_thrust).
    """
    out = {"impact_v_no_thrust": math.sqrt(2.0 * g * max(height, 0.0))}
    if 2.0 * g * (height - h_t) >= v_t ** 2:            # meets the braking profile above h_t
        h_b = (2.0 * g * height - v_t ** 2 + 2.0 * a * h_t) / (2.0 * (g + a))
        v0 = math.sqrt(2.0 * g * (height - h_b))
        hold = mass * g * h_t / v_t if v_t > 0 else 0.0
        imp = mass * (v0 - v_t) * (1.0 + g / a) + hold
        out.update(I=imp, h_b=h_b, v0=v0, t_brake=(v0 - v_t) / a,
                   thrust_brake=mass * (g + a))
    else:                                               # reaches h_t slower than v_t
        h_reach = height - v_t ** 2 / (2.0 * g)         # height where free-fall speed = v_t
        hold_h = max(min(h_reach, h_t), 0.0)
        imp = mass * g * hold_h / v_t if (v_t > 0 and hold_h > 0) else 0.0
        out.update(I=imp, h_b=hold_h, v0=min(v_t, out["impact_v_no_thrust"]), t_brake=0.0,
                   thrust_brake=mass * g if imp > 0 else 0.0)
    return out


def jump_impulse(mass: float, rise: float, thrust: float, g: float = 9.81,
                 leg_credit: float = 0.0) -> dict:
    """Thrust-only boosted-jump estimate (paper section T2): take-off speed v = sqrt(2 g rise),
    net acceleration a = T/m - g, burn t = v / a, impulse I = T t (minus the leg-push credit)."""
    v = math.sqrt(2.0 * g * rise)
    a_net = thrust / mass - g
    if a_net <= 0:
        raise ValueError("thrust does not exceed weight")
    t = v / a_net
    return {"v_takeoff": v, "a_net": a_net, "t_burn": t, "I_thrust_only": thrust * t,
            "I": max(thrust * t - leg_credit, 0.0)}
