"""Two-lane Nagel-Schreckenberg with CWS lane-changing.

Extends the single-lane NaSch model in :mod:`nasch` to two lanes using the
Chowdhury-Wolf-Schreckenberg (CWS) symmetric lane-change rule. The
:class:`TwoLaneNaSchWithLights` subclass reuses the single-lane
:class:`nasch.TrafficCAWithLights` verbatim as a schedule oracle so the
light-timing logic is never duplicated.

Contents
--------
* ``TwoLaneNaSch``                — periodic and open boundaries.
* ``TwoLaneNaSchWithLights``      — adds N equidistant lights spanning both lanes.
* ``two_lane_fundamental_diagram``— flow-vs-density sweep for periodic runs.
"""

import numpy as np

from nasch import TrafficCAWithLights


# ── Vectorised gap helpers ─────────────────────────────────────────────────
# Replace the per-car Python loops that originally dominated runtime. Each
# helper assumes `obstacles` is a sorted ascending int array of cell indices.
# `positions` is also int but does not need to be sorted. Output shape matches
# `positions`. Behaviour is bit-identical to the original loops so the
# regression test (`Bakri/test_two_lane_regression.py`) still passes.

def _gap_ahead_periodic(positions, obstacles, L):
    """Distance to nearest obstacle strictly ahead (with wrap)."""
    if len(obstacles) == 0:
        return np.full(len(positions), L, dtype=int)
    idx = np.searchsorted(obstacles, positions, side="right")
    next_obs = obstacles[idx % len(obstacles)]
    return (next_obs - positions - 1) % L


def _gap_ahead_open(positions, obstacles, L):
    """Distance to nearest obstacle strictly ahead, or L-pos if none."""
    if len(obstacles) == 0:
        return L - positions
    idx = np.searchsorted(obstacles, positions, side="right")
    has_next = idx < len(obstacles)
    next_obs = obstacles[np.minimum(idx, len(obstacles) - 1)]
    return np.where(has_next, next_obs - positions - 1, L - positions)


def _gap_behind_periodic(positions, obstacles, L):
    """Distance to nearest obstacle strictly behind (with wrap)."""
    if len(obstacles) == 0:
        return np.full(len(positions), L, dtype=int)
    idx = np.searchsorted(obstacles, positions, side="left")
    prev_idx = (idx - 1) % len(obstacles)
    prev_obs = obstacles[prev_idx]
    return (positions - prev_obs - 1) % L


def _gap_behind_open(positions, obstacles, L):
    """Distance to nearest obstacle strictly behind, or pos if none."""
    if len(obstacles) == 0:
        return positions.copy()
    idx = np.searchsorted(obstacles, positions, side="left")
    has_prev = idx > 0
    prev_obs = obstacles[np.maximum(idx - 1, 0)]
    return np.where(has_prev, positions - prev_obs - 1, positions)


class TwoLaneNaSch:
    """Two-lane NaSch CA with CWS lane-changing.

    Parameters
    ----------
    L         : road length (cells per lane)
    n_lanes   : 1 or 2. 1 disables lane-change (used for the regression test).
    v_max     : velocity cap. Retains its meaning as the GLOBAL ceiling used
                by lane-change condition (iv) regardless of heterogeneity.
    p_rand    : dawdling probability
    p_chg     : lane-change probability (applied once all four CWS
                conditions are satisfied)
    boundary  : "periodic" (closed, fixed number of cars) or "open"
                (inflow/outflow at cell 0 / cell L)
    n_cars    : required for periodic boundary; total cars across all lanes
    p_in      : inflow probability per lane per step (open boundary only)
    v_max_slow, v_max_fast : per-type ceilings for heterogeneous traffic.
                Both None means homogeneous (every car uses ``v_max``); both
                given activates heterogeneity (must satisfy
                0 <= v_max_slow <= v_max_fast <= v_max). Specifying exactly
                one is a ValueError.
    f_slow    : Bernoulli probability that a freshly placed/spawned car is
                of the slow type. With ``f_slow=0.0`` the type draw is
                short-circuited so the RNG sequence matches the homogeneous
                implementation bit-for-bit.
    """

    def __init__(
        self,
        L=500,
        n_lanes=2,
        v_max=5,
        p_rand=0.3,
        p_chg=1.0,
        boundary="periodic",
        n_cars=None,
        p_in=0.5,
        v_max_slow=None,
        v_max_fast=None,
        f_slow=0.0,
    ):
        if n_lanes not in (1, 2):
            raise ValueError(
                "n_lanes must be 1 or 2; CWS lane-change is only defined "
                "for two lanes."
            )
        if boundary not in ("periodic", "open"):
            raise ValueError("boundary must be 'periodic' or 'open'")
        if boundary == "periodic" and n_cars is None:
            raise ValueError("n_cars is required when boundary='periodic'")

        # Heterogeneity validation.
        if (v_max_slow is None) != (v_max_fast is None):
            raise ValueError(
                "v_max_slow and v_max_fast must both be specified or both None"
            )
        if v_max_slow is None:
            self._heterogeneous = False
            v_max_slow = v_max
            v_max_fast = v_max
        else:
            self._heterogeneous = True
            if not (0 <= v_max_slow <= v_max_fast <= v_max):
                raise ValueError(
                    "Require 0 <= v_max_slow <= v_max_fast <= v_max; got "
                    f"v_max_slow={v_max_slow}, v_max_fast={v_max_fast}, "
                    f"v_max={v_max}"
                )
        if not (0.0 <= f_slow <= 1.0):
            raise ValueError(f"f_slow must be in [0, 1]; got {f_slow}")

        self.L = L
        self.n_lanes = n_lanes
        self.v_max = v_max
        self.p_rand = p_rand
        self.p_chg = p_chg
        self.boundary = boundary
        self.p_in = p_in
        self.v_max_slow = v_max_slow
        self.v_max_fast = v_max_fast
        self.f_slow = f_slow

        self.road = np.full((n_lanes, L), -1, dtype=int)
        self.car_ids = np.full((n_lanes, L), -1, dtype=int)
        self.v_max_arr = np.full((n_lanes, L), -1, dtype=int)
        self.time = 0
        self.outflow_count = 0

        if boundary == "periodic":
            total_cells = n_lanes * L
            if n_cars > total_cells:
                raise ValueError("n_cars exceeds total cells")
            flat = np.random.choice(total_cells, n_cars, replace=False)
            lanes = flat // L
            cols = flat % L
            velocities = np.random.randint(0, v_max + 1, n_cars)
            self.road[lanes, cols] = velocities
            self.car_ids[lanes, cols] = np.arange(n_cars)
            if self.f_slow == 0.0:
                # Short-circuit: no extra RNG draw in homogeneous mode.
                self.v_max_arr[lanes, cols] = self.v_max_fast
            else:
                slow_mask = np.random.random(n_cars) < self.f_slow
                self.v_max_arr[lanes, cols] = np.where(
                    slow_mask, self.v_max_slow, self.v_max_fast
                )
            self.next_id = n_cars
        else:
            self.next_id = 0

    # ── Lane-change pre-step (CWS) ──────────────────────────────────────

    def _gap_ahead_row(self, row, positions):
        """For each pos, distance to next car ahead in `row` (periodic/open)."""
        obstacles = np.where(row >= 0)[0]
        if self.boundary == "periodic":
            return _gap_ahead_periodic(positions, obstacles, self.L)
        return _gap_ahead_open(positions, obstacles, self.L)

    def _gap_behind_row(self, row, positions):
        """For each pos, distance to nearest car *behind* in `row`."""
        obstacles = np.where(row >= 0)[0]
        if self.boundary == "periodic":
            return _gap_behind_periodic(positions, obstacles, self.L)
        return _gap_behind_open(positions, obstacles, self.L)

    def _lane_change_substep(self):
        """Apply CWS symmetric lane-change rule in parallel."""
        new_road = self.road.copy()
        new_ids = self.car_ids.copy()
        new_v_max = self.v_max_arr.copy()

        for src in (0, 1):
            dst = 1 - src
            car_cols = np.where(self.road[src] >= 0)[0]
            if len(car_cols) == 0:
                continue

            vels = self.road[src, car_cols]
            v_max_per_car = self.v_max_arr[src, car_cols]
            gap_own = self._gap_ahead_row(self.road[src], car_cols)
            gap_other_ahead = self._gap_ahead_row(self.road[dst], car_cols)
            gap_other_behind = self._gap_behind_row(self.road[dst], car_cols)
            adj_empty = self.road[dst, car_cols] < 0

            can = (
                (gap_own < np.minimum(vels + 1, v_max_per_car))
                & (gap_other_ahead > gap_own)
                & adj_empty
                & (gap_other_behind > self.v_max)
            )
            roll = np.random.random(len(car_cols)) < self.p_chg
            switch = can & roll

            switch_cols = car_cols[switch]
            new_road[src, switch_cols] = -1
            new_ids[src, switch_cols] = -1
            new_v_max[src, switch_cols] = -1
            new_road[dst, switch_cols] = vels[switch]
            new_ids[dst, switch_cols] = self.car_ids[src, switch_cols]
            new_v_max[dst, switch_cols] = self.v_max_arr[src, switch_cols]

        self.road = new_road
        self.car_ids = new_ids
        self.v_max_arr = new_v_max

    # ── Longitudinal update (per lane, same 4-rule recipe as single-lane) ──

    def _red_light_positions(self):
        """Base class: no lights. Subclass overrides."""
        return np.array([], dtype=int)

    def _step_lane_periodic(self, lane, red_cols):
        row = self.road[lane]
        ids = self.car_ids[lane]
        v_max_lane = self.v_max_arr[lane]
        L = self.L
        p_rand = self.p_rand

        car_pos = np.where(row >= 0)[0]
        if len(car_pos) == 0:
            return

        # Obstacles in this lane = cars + red lights (lights span both lanes)
        if len(red_cols):
            obstacles = np.sort(np.concatenate([car_pos, red_cols]))
        else:
            obstacles = car_pos  # already sorted ascending from np.where

        velocities = row[car_pos].copy()
        v_max_per_car = v_max_lane[car_pos]
        velocities = np.minimum(velocities + 1, v_max_per_car)

        gaps = _gap_ahead_periodic(car_pos, obstacles, L)
        velocities = np.minimum(velocities, gaps)
        velocities = np.maximum(velocities, 0)

        rand_mask = np.random.random(len(car_pos)) < p_rand
        velocities[rand_mask] = np.maximum(velocities[rand_mask] - 1, 0)

        new_row = np.full(L, -1, dtype=int)
        new_ids = np.full(L, -1, dtype=int)
        new_v_max = np.full(L, -1, dtype=int)
        new_pos = (car_pos + velocities) % L
        new_row[new_pos] = velocities
        new_ids[new_pos] = ids[car_pos]
        new_v_max[new_pos] = v_max_per_car

        self.road[lane] = new_row
        self.car_ids[lane] = new_ids
        self.v_max_arr[lane] = new_v_max

    def _step_lane_open(self, lane, red_cols, record_flow):
        row = self.road[lane]
        ids = self.car_ids[lane]
        v_max_lane = self.v_max_arr[lane]
        L = self.L
        p_rand = self.p_rand

        car_pos = np.where(row >= 0)[0]
        if len(car_pos) == 0:
            return

        if len(red_cols):
            obstacles = np.sort(np.concatenate([car_pos, red_cols]))
        else:
            obstacles = car_pos

        velocities = row[car_pos].copy()
        v_max_per_car = v_max_lane[car_pos]
        velocities = np.minimum(velocities + 1, v_max_per_car)

        gaps = _gap_ahead_open(car_pos, obstacles, L)
        velocities = np.minimum(velocities, gaps)
        velocities = np.maximum(velocities, 0)

        rand_mask = np.random.random(len(car_pos)) < p_rand
        velocities[rand_mask] = np.maximum(velocities[rand_mask] - 1, 0)

        new_row = np.full(L, -1, dtype=int)
        new_ids = np.full(L, -1, dtype=int)
        new_v_max = np.full(L, -1, dtype=int)
        new_pos = car_pos + velocities
        exited = new_pos >= L
        if record_flow:
            self.outflow_count += int(np.sum(exited))
        staying = ~exited
        new_row[new_pos[staying]] = velocities[staying]
        new_ids[new_pos[staying]] = ids[car_pos][staying]
        new_v_max[new_pos[staying]] = v_max_per_car[staying]

        self.road[lane] = new_row
        self.car_ids[lane] = new_ids
        self.v_max_arr[lane] = new_v_max

    def _inflow(self):
        for lane in range(self.n_lanes):
            if self.road[lane, 0] < 0 and np.random.random() < self.p_in:
                if self.f_slow == 0.0:
                    # Short-circuit: no extra RNG draw in homogeneous mode.
                    v_max_i = self.v_max_fast
                else:
                    v_max_i = (
                        self.v_max_slow
                        if np.random.random() < self.f_slow
                        else self.v_max_fast
                    )
                self.road[lane, 0] = 0
                self.car_ids[lane, 0] = self.next_id
                self.v_max_arr[lane, 0] = v_max_i
                self.next_id += 1

    # ── Public step / runners ───────────────────────────────────────────

    def step(self, record_flow=False):
        if self.n_lanes == 2 and self.p_chg > 0:
            self._lane_change_substep()

        red_cols = self._red_light_positions()
        for lane in range(self.n_lanes):
            if self.boundary == "periodic":
                self._step_lane_periodic(lane, red_cols)
            else:
                self._step_lane_open(lane, red_cols, record_flow)

        if self.boundary == "open":
            self._inflow()

        self.time += 1

    def warmup(self, T):
        for _ in range(T):
            self.step(record_flow=False)
        self.outflow_count = 0

    def run(self, T):
        self.outflow_count = 0
        for _ in range(T):
            self.step(record_flow=True)
        return self.outflow_count / T

    def mean_flow(self):
        """Flow density J = <v> * rho, averaged over all (n_lanes, L) cells.

        For n_lanes=1 this matches `TrafficCA_Periodic.mean_flow()` exactly.
        """
        mask = self.road >= 0
        if not np.any(mask):
            return 0.0
        total_v = float(np.sum(self.road[mask]))
        return total_v / (self.n_lanes * self.L)

    def density(self):
        return float(np.sum(self.road >= 0)) / (self.n_lanes * self.L)

    def get_snapshot(self):
        """Binary occupancy array of shape (n_lanes, L)."""
        return (self.road >= 0).astype(int)

class TwoLaneNaSchWithLights(TwoLaneNaSch):
    """Two-lane NaSch with N evenly-spaced lights spanning both lanes.

    Light timing is delegated to a private `TrafficCAWithLights` instance
    (imported from the notebook via `_nasch_nb`) used as a schedule oracle:
    the single-lane code is the authoritative source for the
    `(N, T_cycle, T_green, offset)` → red-columns mapping, and this class
    does not re-implement it.

    Phase scheduling accepts two mutually complementary inputs:
    * ``offset`` (bool) — evenly-spaced phases of ``T_cycle/N`` between
      lights when True; all lights share phase 0 when False.
    * ``phase_offsets`` (array-like of length N, optional) — explicit
      per-light phase array, overriding whatever ``offset`` produced.
      Leave as None to keep the original ``offset`` behaviour.
    """

    def __init__(
        self,
        L=500,
        n_lanes=2,
        v_max=5,
        p_rand=0.3,
        p_chg=1.0,
        boundary="open",
        n_cars=None,
        p_in=0.5,
        N=2,
        T_cycle=30,
        T_green=None,
        offset=False,
        phase_offsets=None,
        v_max_slow=None,
        v_max_fast=None,
        f_slow=0.0,
    ):
        super().__init__(
            L=L,
            n_lanes=n_lanes,
            v_max=v_max,
            p_rand=p_rand,
            p_chg=p_chg,
            boundary=boundary,
            n_cars=n_cars,
            p_in=p_in,
            v_max_slow=v_max_slow,
            v_max_fast=v_max_fast,
            f_slow=f_slow,
        )
        self._light_oracle = TrafficCAWithLights(
            L=L, N=N, T_cycle=T_cycle, T_green=T_green, offset=offset
        )
        if phase_offsets is not None:
            phase_offsets = np.asarray(phase_offsets, dtype=int)
            if phase_offsets.shape != (N,):
                raise ValueError(
                    f"phase_offsets must have shape ({N},), got {phase_offsets.shape}"
                )
            self._light_oracle.phase_offsets = phase_offsets
        self.N = N
        self.T_cycle = T_cycle
        self.T_green = self._light_oracle.T_green
        self.offset = offset
        self.light_pos = self._light_oracle.light_pos
        self.phase_offsets = self._light_oracle.phase_offsets

    def _red_light_positions(self):
        self._light_oracle.time = self.time
        return self._light_oracle._red_light_positions()

    def light_states(self):
        self._light_oracle.time = self.time
        return self._light_oracle.light_states()

def two_lane_fundamental_diagram(v_max=5, L=200, p_rand=0.3, n_lanes=1,
                                  p_chg=0.0, n_densities=40,
                                  T_warmup=300, T_measure=500):
    """Compute J vs rho for the periodic two-lane NaSch model.

    Density rho is cars per cell (total cars / (n_lanes * L)).
    """
    densities = np.linspace(0.01, 0.99, n_densities)
    flows     = np.zeros(n_densities)
    for k, rho in enumerate(densities):
        n_cars = max(1, int(rho * n_lanes * L))
        sim = TwoLaneNaSch(L=L, n_lanes=n_lanes, v_max=v_max, p_rand=p_rand,
                           p_chg=p_chg, boundary='periodic', n_cars=n_cars)
        for _ in range(T_warmup):
            sim.step()
        J = 0.0
        for _ in range(T_measure):
            sim.step()
            J += sim.mean_flow()
        flows[k] = J / T_measure
    return densities, flows
