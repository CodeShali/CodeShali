"""Thruster bank: first-order-lag thrusters with shared consumable budgets.

Model (per physics step ``dt``, vectorised over all N thrusters):

1. Command clamp: ``u = clip(f_cmd, 0, f_max_eff)`` (push only).
2. First-order lag, forward Euler (spec formula)::

       f += (u - f) * alpha,   alpha = min(dt / tau_eff, 1)

   The ``min(.,1)`` keeps the update unconditionally stable when a randomised tau drops
   below dt (alpha = 1 means "no lag"). With dt = 1 ms and tau = 5 ms the Euler response
   after one tau is 1 - (1 - 0.2)^5 = 67.2 % (vs. 63.2 % for the continuous system).
3. Output clamp to ``[0, f_max_eff]``.
4. Budget deduction:
   * gas budgets (``impulse_gas_pods`` / ``impulse_gas_jets``): impulse ``sum(f) * dt``
     from the class's shared tank;
   * ``battery_wh`` (fans): ``sum(k f^1.5) * dt / 3600`` with ``k = P_peak / f_max^1.5``
     fitted on the NOMINAL f_max from the config (domain randomisation does not change k).
   If a step would overdraw a budget, all forces on that budget are scaled uniformly so the
   budget lands exactly at 0 (impulse is linear in f -> s = rem/demand; fan energy goes
   like f^1.5 -> s = (rem/demand)^(2/3)). Once a budget is 0, every thruster on it outputs 0
   and its ``u_max`` is 0.

Hardware-realistic options (per actuator class in ``suit.actuator_classes``; absent = the
continuous first-order-lag model above, so the default suit is unchanged):

``valve: {type: onoff, min_on_time, min_off_time, open_delay, close_delay, rise_tau,
threshold}`` — solenoid (cold-gas) valve.  The thruster is either 0 or ``f_max``.  The
continuous command ``u`` is turned into a valve command by a per-thruster sigma-delta
modulator at the physics rate: ``e += (u - f_max * s) dt`` (``s`` = commanded valve state);
the valve is commanded open when ``e > threshold`` and closed when the error after this
step would fall below ``-threshold`` (default threshold = half a step's impulse ``0.5 f_max
dt``, i.e. rounding; a wider hysteresis only widens the limit cycle), never before
``min_on_time`` / ``min_off_time`` have elapsed.  With the realistic jets (40 N, 5 ms
minimum pulse) every 100 ms window tracks a constant command within ~2 %.  The physical valve follows the
command after ``open_delay`` / ``close_delay`` (quantised to physics steps; a close-then-
reopen gap shorter than ``close_delay - open_delay`` keeps the valve open, a pulse shorter
than ``open_delay - close_delay`` is swallowed), then the force rises with an optional
first-order lag ``rise_tau`` (0 = square pulses; a lag conserves the pulse impulse).  The
modulator subtracts the systematic pulse-width change ``f_max (close_delay - open_delay)``
at every opening, so the mean force tracks ``u``.  Domain randomisation: ``scale_tau``
scales the delays and ``rise_tau``; ``scale_fmax`` scales the on-force.

``solid_pulses: {charges_per_pod, charge_impulse, burn_time, ignition_delay, scheduler,
trigger_fraction, accumulator_limit, max_concurrent}`` — solid gas-generator pulse charges.
Each charge, once ignited, burns after ``ignition_delay`` at ``F = charge_impulse /
burn_time`` for ``burn_time`` and cannot be throttled or stopped.  ``scheduler: auto``
(default) keeps existing controllers working: the continuous command is integrated per pod,
``acc += u dt``, and a charge is ignited (``acc -= charge_impulse``) when ``acc >=
trigger_fraction * charge_impulse`` (``acc`` clamped to ``+-accumulator_limit`` charges).
``scheduler: manual`` ignores the command; the controller calls :meth:`ThrusterBank.ignite`.
A pod burns at most ``max_concurrent`` charges at once; a charge occupies its slot for
``burn_time`` from the ignition command, so the next one can be pre-ignited to start as the
previous ends.  Forces are integrated exactly over each physics step (impulse per charge is
exact for any dt).  The budget of a solid-pulse class is its charges (``charges_per_pod *
charge_impulse * n_pods``, see :func:`suitsim.interfaces.budget_capacity`).  Domain
randomisation: ``scale_fmax`` scales the burn thrust at constant charge impulse (burn time
``/ scale``), ``scale_tau`` scales the ignition delay.
"""
from __future__ import annotations

import numpy as np

from suitsim.interfaces import BUDGET_KEYS, THRUSTER_CLASSES, ThrusterSpec, budget_capacity

_SEC_PER_HOUR = 3600.0
_BATTERY_KEY = "battery_wh"

_VALVE_DEFAULTS = {
    "min_on_time": 0.0,        # [s]
    "min_off_time": 0.0,       # [s]
    "open_delay": 0.0,         # [s] command -> valve open
    "close_delay": 0.0,        # [s] command -> valve closed
    "rise_tau": 0.0,           # [s] first-order force rise/decay after the valve moves (0 = none)
    "threshold": None,         # [N*s] modulator hysteresis; None = 0.5 * f_max * dt
}
_PULSE_DEFAULTS = {
    "ignition_delay": 0.0,     # [s]
    "scheduler": "auto",       # auto | manual
    "trigger_fraction": 0.5,   # ignite when the commanded impulse backlog reaches this x charge
    "accumulator_limit": 1.0,  # [charges] clamp of the commanded-impulse accumulator (+-)
    "max_concurrent": 1,       # charges burning at once per pod
}


def fan_k(cfg: dict) -> float:
    """Fan power coefficient k [W / N^1.5] so that P(f_max) == fan_peak_power_w.

    0 when the suit has no fan class (realistic variants)."""
    fan = (cfg["suit"]["actuator_classes"] or {}).get("fan")
    if not fan or int(fan.get("count", 1)) == 0:
        return 0.0
    p_peak = float(cfg["suit"]["budgets"]["fan_peak_power_w"])
    return p_peak / float(fan["f_max"]) ** 1.5


def fan_power(f: np.ndarray | float, k: float | dict) -> np.ndarray | float:
    """Electrical power [W] of a fan producing thrust ``f`` [N]: ``P = k f^1.5``.

    ``k`` is either the coefficient itself or a config dict (then :func:`fan_k` is used).
    """
    if isinstance(k, dict):
        k = fan_k(k)
    f = np.maximum(np.asarray(f, dtype=float), 0.0)
    p = k * f ** 1.5
    return float(p) if p.ndim == 0 else p


class ThrusterBank:
    """Implements :class:`suitsim.interfaces.ThrusterBankProtocol`."""

    def __init__(self, specs: list[ThrusterSpec], cfg: dict, dt: float):
        self.specs = list(specs)
        self.n = len(self.specs)
        self.dt = float(dt)
        self.names = [s.name for s in self.specs]

        self.f_max_nom = np.array([s.f_max for s in self.specs], dtype=float)
        self.tau_nom = np.array([s.tau for s in self.specs], dtype=float)
        cls = np.array([s.cls for s in self.specs])
        self._cls_idx = {c: np.flatnonzero(cls == c) for c in THRUSTER_CLASSES}
        self.idx_pods = self._cls_idx["lift_pod"]
        self.idx_fans = self._cls_idx["fan"]
        self.idx_jets = self._cls_idx["micro_jet"]

        self.capacity = {k: budget_capacity(cfg, k) for k in BUDGET_KEYS}
        bud = np.array([s.budget for s in self.specs])
        unknown = set(bud) - set(BUDGET_KEYS)
        if unknown:
            raise ValueError(f"unknown budget keys in thruster specs: {unknown}")
        self._bud_idx = {k: np.flatnonzero(bud == k) for k in BUDGET_KEYS}
        self._gas_keys = tuple(k for k in BUDGET_KEYS if k != _BATTERY_KEY)
        self.k_fan = fan_k(cfg)

        self._bud_of = bud
        self._setup_valves(cfg)
        self._setup_pulses(cfg)
        self.reset()

    def _class_cfg(self, cfg: dict, key: str) -> dict:
        classes = cfg["suit"]["actuator_classes"]
        return {c: classes[c][key] for c in {s.cls for s in self.specs}
                if classes.get(c) and classes[c].get(key)}

    def _setup_valves(self, cfg: dict) -> None:
        vc = self._class_cfg(cfg, "valve")
        for c, v in vc.items():
            if str(v.get("type", "onoff")) != "onoff":
                raise ValueError(f"actuator class {c!r}: unknown valve type {v.get('type')!r}")
        self.idx_valve = np.array([i for i, s in enumerate(self.specs) if s.cls in vc], int)
        pv = [{**_VALVE_DEFAULTS, **vc[self.specs[i].cls]} for i in self.idx_valve]
        dt = self.dt
        self._v_min_on = np.array([int(round(float(p["min_on_time"]) / dt)) for p in pv], int)
        self._v_min_off = np.array([int(round(float(p["min_off_time"]) / dt)) for p in pv], int)
        self._v_open_nom = np.array([float(p["open_delay"]) for p in pv])
        self._v_close_nom = np.array([float(p["close_delay"]) for p in pv])
        self._v_rise_nom = np.array([float(p["rise_tau"]) for p in pv])
        self._v_thresh_cfg = np.array([np.nan if p["threshold"] is None else float(p["threshold"])
                                       for p in pv])

    def _setup_pulses(self, cfg: dict) -> None:
        pc = self._class_cfg(cfg, "solid_pulses")
        self.idx_pulse = np.array([i for i, s in enumerate(self.specs) if s.cls in pc], int)
        pp = [{**_PULSE_DEFAULTS, **pc[self.specs[i].cls]} for i in self.idx_pulse]
        for p in pp:
            if str(p["scheduler"]) not in ("auto", "manual"):
                raise ValueError(f"solid_pulses.scheduler must be auto|manual, got {p['scheduler']!r}")
        self._p_charges = np.array([int(p["charges_per_pod"]) for p in pp], int)
        self._p_impulse = np.array([float(p["charge_impulse"]) for p in pp])
        self._p_burn_nom = np.array([float(p["burn_time"]) for p in pp])
        self._p_delay_nom = np.array([float(p["ignition_delay"]) for p in pp])
        self._p_auto = np.array([str(p["scheduler"]) == "auto" for p in pp], bool)
        self._p_trigger = np.array([float(p["trigger_fraction"]) for p in pp])
        self._p_acc_lim = np.array([float(p["accumulator_limit"]) for p in pp])
        self._p_conc = np.array([int(p["max_concurrent"]) for p in pp], int)
        self._pulse_budgets = {self.specs[i].budget for i in self.idx_pulse}
        self._mask_cont = np.ones(self.n, bool)
        self._mask_cont[self.idx_valve] = False
        self._mask_cont[self.idx_pulse] = False

    # ------------------------------------------------------------------ setup
    def class_index(self, cls: str) -> np.ndarray:
        return self._cls_idx[cls]

    def budget_index(self, key: str) -> np.ndarray:
        return self._bud_idx[key]

    def reset(self, scale_fmax: np.ndarray | None = None,
              scale_tau: np.ndarray | None = None) -> None:
        """Zero forces, refill budgets, apply per-thruster multiplicative DR scales."""
        sf = np.ones(self.n) if scale_fmax is None else np.broadcast_to(
            np.asarray(scale_fmax, dtype=float), (self.n,))
        st = np.ones(self.n) if scale_tau is None else np.broadcast_to(
            np.asarray(scale_tau, dtype=float), (self.n,))
        if np.any(sf < 0) or np.any(st <= 0):
            raise ValueError("scale_fmax must be >= 0 and scale_tau > 0")
        self.f_max = self.f_max_nom * sf
        self.tau = self.tau_nom * st
        self.alpha = np.minimum(self.dt / self.tau, 1.0)
        self._f = np.zeros(self.n)
        self._cmd = np.zeros(self.n)
        self._rem = dict(self.capacity)
        self._alive = np.ones(self.n, dtype=bool)
        self._fan_power = np.zeros(len(self.idx_fans))
        self._t = 0.0          # bank time [s] (start of the next step)
        self._k = 0            # physics steps taken
        if self.idx_valve.size:
            iv, dt = self.idx_valve, self.dt
            stv = st[iv]
            self._v_nopen = np.rint(self._v_open_nom * stv / dt).astype(int)
            self._v_nclose = np.rint(self._v_close_nom * stv / dt).astype(int)
            rise = self._v_rise_nom * stv
            self._v_alpha = np.where(rise > 0, np.minimum(dt / np.maximum(rise, 1e-300), 1.0), 1.0)
            lo = np.minimum(self._v_nopen, self._v_nclose)
            hi = np.maximum(self._v_nopen, self._v_nclose)
            H = int(hi.max()) + 1 if hi.size else 1
            lags = np.arange(H)
            self._v_win = (lags[None, :] >= lo[:, None]) & (lags[None, :] <= hi[:, None])
            self._v_dilate = self._v_nclose >= self._v_nopen      # OR window, else AND window
            self._v_hist = np.zeros((iv.size, H), bool)          # lag-ordered command history
            self._v_cmd = np.zeros(iv.size, bool)                # commanded valve state
            self._v_phys = np.zeros(iv.size, bool)               # physical valve state
            self._v_count = np.full(iv.size, 1 << 30)            # steps in the current state
            self._v_err = np.zeros(iv.size)                      # modulator error [N*s]
            self._v_f = np.zeros(iv.size)
            fm = self.f_max[iv]
            th = np.where(np.isnan(self._v_thresh_cfg), 0.5 * fm * dt, self._v_thresh_cfg)
            self._v_thresh = th
            # systematic pulse-width change of the delays, removed at every opening
            self._v_width_comp = fm * (self._v_nclose - self._v_nopen) * dt
            self._v_pulses = np.zeros(iv.size, int)
        if self.idx_pulse.size:
            ip = self.idx_pulse
            self._p_force = self._p_impulse / self._p_burn_nom * sf[ip]   # burn thrust [N]
            self._p_burn = self._p_impulse / np.maximum(self._p_force, 1e-300)
            self._p_delay = self._p_delay_nom * st[ip]
            self._p_left = self._p_charges.copy()
            self._p_acc = np.zeros(ip.size)
            self._p_active: list[list[tuple[float, float, float]]] = [[] for _ in ip]  # (t_cmd, ts, te)
            self._p_log: list[tuple[float, int]] = []                           # (t_cmd, pod)

    # ------------------------------------------------------------------ solid pulses
    def ignite(self, pod_index: int) -> bool:
        """Ignite the next charge of solid-pulse thruster ``pod_index`` (index into
        :attr:`idx_pulse`, i.e. the pulse pods in spec order) at the start of the next physics
        step.  Returns False (nothing happens) when the pod has no charge left or already
        burns ``max_concurrent`` charges.  In ``auto`` mode the charge also counts against the
        scheduler's accumulator."""
        if not self.idx_pulse.size:
            raise RuntimeError("no solid-pulse thrusters in this bank")
        j = int(pod_index)
        if not 0 <= j < self.idx_pulse.size:
            raise IndexError(f"pod_index {j} out of range (0..{self.idx_pulse.size - 1})")
        return self._try_ignite(j, self._t)

    def _try_ignite(self, j: int, t: float) -> bool:
        act = self._p_active[j]
        busy = sum(1 for (tc, _, _) in act if t < tc + self._p_burn[j] - 1e-12)
        if self._p_left[j] <= 0 or busy >= self._p_conc[j]:
            return False
        ts = self._snap(t + self._p_delay[j])
        act.append((t, ts, self._snap(ts + self._p_burn[j])))
        self._p_left[j] -= 1
        self._p_log.append((t, j))
        if self._p_auto[j]:
            self._p_acc[j] -= self._p_impulse[j]
        return True

    def _snap(self, x: float) -> float:
        """Snap an event time to the physics-step grid when it is on it up to float noise
        (so a charge whose times are whole steps never leaves a sliver force)."""
        k = round(x / self.dt)
        return k * self.dt if abs(x / self.dt - k) < 1e-6 else x

    def _pulse_step(self, cmd: np.ndarray) -> np.ndarray:
        """Forces (n_pulse,) of the solid-pulse thrusters over [t, t+dt)."""
        t, dt = self._t, self.dt
        for j in range(self.idx_pulse.size):
            if self._p_auto[j]:
                lim = self._p_acc_lim[j] * self._p_impulse[j]
                self._p_acc[j] = min(self._p_acc[j] + cmd[j] * dt, lim)
                if self._p_acc[j] >= self._p_trigger[j] * self._p_impulse[j]:
                    self._try_ignite(j, t)
                self._p_acc[j] = max(self._p_acc[j], -lim)
        f = np.zeros(self.idx_pulse.size)
        t1 = (self._k + 1) * dt          # step bounds on the integer grid (see _snap)
        for j, act in enumerate(self._p_active):
            if not act:
                continue
            keep = []
            for (tc, ts, te) in act:
                ov = min(te, t1) - max(ts, t)
                if ov > 0:
                    f[j] += self._p_force[j] * ov / dt
                if te > t1 + 1e-12:
                    keep.append((tc, ts, te))
            self._p_active[j] = keep
        return f

    # ------------------------------------------------------------------ valves
    def _valve_step(self, cmd: np.ndarray) -> np.ndarray:
        """Forces (n_valve,) of the on/off-valve thrusters over [t, t+dt)."""
        dt = self.dt
        fm = self.f_max[self.idx_valve]
        e = self._v_err + cmd * dt                                # error before this step's output
        on = self._v_cmd
        cnt = self._v_count
        turn_on = (~on) & (cnt >= self._v_min_off) & (e > self._v_thresh) & (fm > 0)
        turn_off = on & (cnt >= self._v_min_on) & ((e - fm * dt) < -self._v_thresh)
        new = (on | turn_on) & ~turn_off
        changed = new != on
        self._v_count = np.where(changed, 1, cnt + 1)
        e = e - fm * dt * new - self._v_width_comp * turn_on
        self._v_err = e
        self._v_cmd = new
        self._v_pulses += turn_on
        hist = self._v_hist
        hist[:, 1:] = hist[:, :-1]
        hist[:, 0] = new
        w = self._v_win
        phys = np.where(self._v_dilate, (hist & w).any(axis=1), (hist | ~w).all(axis=1))
        self._v_phys = phys
        self._v_f += (fm * phys - self._v_f) * self._v_alpha
        return self._v_f.copy()

    # ------------------------------------------------------------------ step
    def step(self, f_cmd: np.ndarray) -> np.ndarray:
        f_cmd = np.asarray(f_cmd, dtype=float)
        if f_cmd.shape != (self.n,):
            raise ValueError(f"f_cmd must have shape ({self.n},), got {f_cmd.shape}")
        u_max = self.u_max
        cmd = np.clip(np.nan_to_num(f_cmd, nan=0.0), 0.0, u_max)
        self._cmd = cmd
        f = self._f + (cmd - self._f) * self.alpha
        if self.idx_valve.size:
            f[self.idx_valve] = self._valve_step(cmd[self.idx_valve])
        np.clip(f, 0.0, u_max, out=f)
        if self.idx_pulse.size:
            # burning charges cannot be stopped: never clipped (budget = the charges)
            f[self.idx_pulse] = self._pulse_step(cmd[self.idx_pulse])

        dt = self.dt
        for key in self._gas_keys:
            idx = self._bud_idx[key]
            if idx.size == 0:
                continue
            demand = f[idx].sum() * dt
            rem = self._rem[key]
            if key in self._pulse_budgets:
                self._rem[key] = max(rem - demand, 0.0)
                continue
            if demand > rem:
                f[idx] *= rem / demand if demand > 0 else 0.0
                self._rem[key] = 0.0
                self._alive[idx] = False
            else:
                self._rem[key] = rem - demand
                if self._rem[key] <= 0.0:
                    self._alive[idx] = False

        idx = self._bud_idx[_BATTERY_KEY]
        if idx.size:
            p = self.k_fan * f[idx] ** 1.5
            demand = p.sum() * dt / _SEC_PER_HOUR
            rem = self._rem[_BATTERY_KEY]
            if demand > rem:
                s = (rem / demand) ** (2.0 / 3.0) if demand > 0 else 0.0
                f[idx] *= s
                p *= s ** 1.5
                self._rem[_BATTERY_KEY] = 0.0
                self._alive[idx] = False
            else:
                self._rem[_BATTERY_KEY] = rem - demand
                if self._rem[_BATTERY_KEY] <= 0.0:
                    self._alive[idx] = False
            # fan-power array is ordered like idx_fans (fans are the battery consumers)
            self._fan_power = self.k_fan * f[self.idx_fans] ** 1.5

        self._f = f
        self._k += 1
        self._t = self._k * dt
        return f.copy()

    # ------------------------------------------------------------------ queries
    @property
    def forces(self) -> np.ndarray:
        return self._f.copy()

    @property
    def commands(self) -> np.ndarray:
        return self._cmd.copy()

    @property
    def u_max(self) -> np.ndarray:
        um = np.where(self._alive, self.f_max, 0.0)
        if self.idx_pulse.size:
            um[self.idx_pulse] = np.where(self._p_left > 0, self.f_max[self.idx_pulse], 0.0)
        return um

    def charges_remaining(self) -> np.ndarray:
        """(n_pulse,) unignited charges per solid-pulse pod (order of idx_pulse)."""
        return self._p_left.copy() if self.idx_pulse.size else np.zeros(0, int)

    def charges_used(self) -> int:
        return int((self._p_charges - self._p_left).sum()) if self.idx_pulse.size else 0

    def ignition_log(self) -> list[tuple[float, int]]:
        """[(t_command [s], pod_index)] of every ignition this episode."""
        return list(self._p_log) if self.idx_pulse.size else []

    def valve_pulses(self) -> int:
        """Number of valve openings commanded this episode (on/off thrusters)."""
        return int(self._v_pulses.sum()) if self.idx_valve.size else 0

    def remaining(self) -> dict[str, float]:
        return {k: float(self._rem[k]) for k in BUDGET_KEYS}

    def used(self) -> dict[str, float]:
        return {k: float(self.capacity[k] - self._rem[k]) for k in BUDGET_KEYS}

    def fan_power_w(self) -> np.ndarray:
        """(n_fans,) current electrical power per fan [W], order of idx_fans."""
        return self._fan_power.copy()

    def telemetry(self) -> dict:
        return {
            "forces": self._f.copy(),
            "commands": self._cmd.copy(),
            "u_max": self.u_max,
            "remaining": self.remaining(),
            "used": self.used(),
            "class_force": {c: float(self._f[i].sum()) for c, i in self._cls_idx.items()},
            "fan_power_w": self._fan_power.copy(),
            "charges_remaining": self.charges_remaining(),
            "valve_open": (self._v_phys.copy() if self.idx_valve.size else np.zeros(0, bool)),
        }
