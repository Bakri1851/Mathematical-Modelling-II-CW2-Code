"""Single-lane Nagel-Schreckenberg NaSch cellular automaton.

This is the authoritative single-lane implementation, shared verbatim with
`notebooks/nagel_schreckenberg.ipynb` — which retains its own inline copy as
the original coursework submission (kept self-contained for submission
integrity). All other notebooks should import from this module.

Contents
--------
* ``TrafficCA``             — open-boundary CA with inflow/outflow.
* ``TrafficCAWithLights``   — as above, with N equidistant traffic lights.
* ``TrafficCA_Periodic``    — closed-boundary CA for fundamental-diagram work.
* ``fundamental_diagram``   — flow-vs-density sweep for the periodic CA.

Module constants ``T_WARMUP`` and ``T_MEASURE`` mirror Section 1 of the
notebook so that ``TrafficCA.warmup()`` / ``.run()`` fall back to the same
defaults when no argument is passed.
"""

import numpy as np

# Defaults used as fallbacks by TrafficCA.warmup() / .run() when T is None
# — values mirror Section 1 of nasch_two_lane.ipynb.
T_WARMUP  = 500
T_MEASURE = 1000


class TrafficCA:
    """
    Nagel-Schreckenberg cellular automaton with open boundary conditions.

    Road state: 1D array of length L.
      cell = -1  → empty
      cell >= 0  → car with velocity v
    A parallel `car_ids` array assigns each injected car a unique integer ID
    that travels with it until it exits the road. This lets us tag and follow
    an individual car through a space-time diagram.
    """

    def __init__(self, L=500, v_max=5, p_rand=0.3, p_in=0.5):
        self.L      = L
        self.v_max  = v_max
        self.p_rand = p_rand
        self.p_in   = p_in
        self.road    = np.full(L, -1, dtype=int)  # start empty
        self.car_ids = np.full(L, -1, dtype=int)  # parallel ID array
        self.next_id = 0
        self.time   = 0
        self.outflow_count = 0  # cars that exited during measurement
        self.entry_time   = {}  # car_id -> timestep when injected at cell 0
        self.travel_times = []  # exit_time - entry_time for cars that finished

    def reset(self):
        self.road[:]    = -1
        self.car_ids[:] = -1
        self.next_id    = 0
        self.time       = 0
        self.outflow_count = 0
        self.entry_time.clear()
        self.travel_times = []

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _car_positions(self):
        return np.where(self.road >= 0)[0]

    def _gap_ahead(self, pos):
        """
        Gap (number of empty cells) between car at `pos` and
        the next obstacle (car or road boundary).
        Returns L - pos if no obstacle is found (open road to boundary).
        """
        road = self.road
        gap = 0
        for j in range(pos + 1, self.L):
            if road[j] >= 0:
                return gap
            gap += 1
        return gap  # reached end of road

    def _gap_to_obstacles(self, positions, obstacle_positions):
        """
        Vectorised: for each car position, find distance to next obstacle
        (car or red light). Returns array of gaps.
        `obstacle_positions` is sorted array of all blocking cell indices.
        """
        L = self.L
        if len(obstacle_positions) == 0:
            return L - positions
        idx = np.searchsorted(obstacle_positions, positions, side='right')
        has_next = idx < len(obstacle_positions)
        next_obs = obstacle_positions[np.minimum(idx, len(obstacle_positions) - 1)]
        return np.where(has_next, next_obs - positions - 1, L - positions)

    # ── Red-light obstacle positions (overridden by subclass) ─────────────────

    def _red_light_positions(self):
        """Return array of cell indices where a red light blocks traffic."""
        return np.array([], dtype=int)

    # ── Main update step ──────────────────────────────────────────────────────

    def step(self, record_flow=False, record_travel_times=False):
        road    = self.road
        L       = self.L
        v_max   = self.v_max
        p_rand  = self.p_rand

        car_pos = self._car_positions()
        if len(car_pos) == 0:
            self._inflow()
            self.time += 1
            return

        # Obstacle list = cars + red lights
        red_pos = self._red_light_positions()
        obstacles = np.sort(np.concatenate([car_pos, red_pos]))

        velocities = road[car_pos].copy()

        # Rule 1: Acceleration
        velocities = np.minimum(velocities + 1, v_max)

        # Rule 2: Braking — gap to nearest obstacle ahead
        gaps = self._gap_to_obstacles(car_pos, obstacles)
        velocities = np.minimum(velocities, gaps)
        velocities = np.maximum(velocities, 0)

        # Rule 3: Randomisation
        rand_mask = np.random.random(len(car_pos)) < p_rand
        velocities[rand_mask] = np.maximum(velocities[rand_mask] - 1, 0)

        # Rule 4: Movement — build new road + id state
        new_road = np.full(L, -1, dtype=int)
        new_ids  = np.full(L, -1, dtype=int)
        new_pos  = car_pos + velocities

        exited = new_pos >= L
        if record_flow:
            self.outflow_count += int(np.sum(exited))

        # Drain entry-time bookkeeping for any car that just left the road,
        # appending its travel time during measurement runs.
        if np.any(exited):
            exiting_ids = self.car_ids[car_pos][exited]
            for cid in exiting_ids:
                entered = self.entry_time.pop(int(cid), None)
                if record_travel_times and entered is not None:
                    self.travel_times.append(self.time - entered)

        staying = ~exited
        new_road[new_pos[staying]] = velocities[staying]
        new_ids[new_pos[staying]]  = self.car_ids[car_pos][staying]

        self.road    = new_road
        self.car_ids = new_ids

        # Inflow at cell 0
        self._inflow()
        self.time += 1

    def _inflow(self):
        """Inject a new car at cell 0 with probability p_in if cell is free."""
        if self.road[0] < 0 and np.random.random() < self.p_in:
            self.road[0]    = 0              # new car, velocity 0
            self.car_ids[0] = self.next_id   # assign fresh ID
            self.entry_time[self.next_id] = self.time
            self.next_id   += 1

    # ── Simulation runners ────────────────────────────────────────────────────

    def warmup(self, T=None):
        T = T or T_WARMUP
        for _ in range(T):
            self.step(record_flow=False, record_travel_times=False)
        self.outflow_count = 0
        self.travel_times = []

    def run(self, T=None):
        """Run T measurement steps, return mean flow (cars/timestep)."""
        T = T or T_MEASURE
        self.outflow_count = 0
        self.travel_times = []
        for _ in range(T):
            self.step(record_flow=True, record_travel_times=True)
        return self.outflow_count / T

    def get_snapshot(self):
        """Return binary occupancy array (1=car, 0=empty)."""
        return (self.road >= 0).astype(int)

    def density(self):
        return np.sum(self.road >= 0) / self.L

    def mean_travel_time(self):
        """Mean per-car travel time (entry-to-exit) recorded during run()."""
        return float(np.mean(self.travel_times)) if self.travel_times else float("nan")


class TrafficCAWithLights(TrafficCA):
    """
    NaSch CA extended with N equidistant traffic lights.

    Parameters
    ----------
    N             : number of traffic lights
    T_cycle       : total switching cycle length (timesteps)
    T_green       : number of green timesteps per cycle (T_red = T_cycle - T_green)
    offset        : if True, consecutive lights are phase-shifted by T_cycle/N
    phase_offsets : optional explicit per-light phase array; overrides ``offset``.
                    Used by sensitivity work for green-wave/random strategies.
    """

    def __init__(self, L=500, v_max=5, p_rand=0.3, p_in=0.5,
                 N=2, T_cycle=30, T_green=None, offset=False,
                 phase_offsets=None):
        super().__init__(L=L, v_max=v_max, p_rand=p_rand, p_in=p_in)
        self.N       = N
        self.T_cycle = T_cycle
        self.T_green = T_green if T_green is not None else T_cycle // 2
        self.offset  = offset

        # Light positions — evenly spaced
        self.light_pos = np.array(
            [L * (i + 1) // (N + 1) for i in range(N)], dtype=int
        )

        # Phase offsets per light: explicit array wins, else evenly-spread, else zero.
        if phase_offsets is not None:
            phase_offsets = np.asarray(phase_offsets, dtype=int)
            if phase_offsets.shape != (N,):
                raise ValueError(
                    f"phase_offsets must have shape ({N},), got {phase_offsets.shape}"
                )
            self.phase_offsets = phase_offsets % T_cycle
        elif offset:
            self.phase_offsets = np.array(
                [int(i * T_cycle / N) for i in range(N)], dtype=int
            )
        else:
            self.phase_offsets = np.zeros(N, dtype=int)

    def _light_is_red(self, light_idx):
        """Return True if light `light_idx` is currently red."""
        phase = (self.time + self.phase_offsets[light_idx]) % self.T_cycle
        return phase >= self.T_green  # green phase first, then red

    def _red_light_positions(self):
        """Return positions of all currently red lights."""
        red = [self.light_pos[i] for i in range(self.N) if self._light_is_red(i)]
        return np.array(red, dtype=int)

    def light_states(self):
        """Return dict of {position: 'green'/'red'} for current timestep."""
        return {
            self.light_pos[i]: ('red' if self._light_is_red(i) else 'green')
            for i in range(self.N)
        }


class TrafficCA_Periodic:
    """NaSch CA with periodic (closed) boundary — for fundamental diagram."""

    def __init__(self, L=500, n_cars=100, v_max=5, p_rand=0.3):
        self.L      = L
        self.v_max  = v_max
        self.p_rand = p_rand
        self.road   = np.full(L, -1, dtype=int)
        # Place cars randomly
        positions = np.random.choice(L, n_cars, replace=False)
        self.road[positions] = np.random.randint(0, v_max + 1, n_cars)

    def step(self):
        road   = self.road
        L      = self.L
        v_max  = self.v_max
        p_rand = self.p_rand

        car_pos = np.where(road >= 0)[0]
        if len(car_pos) == 0:
            return

        velocities = road[car_pos].copy()

        # Rule 1: Acceleration
        velocities = np.minimum(velocities + 1, v_max)

        # Rule 2: Braking — periodic gap
        gaps = np.empty(len(car_pos), dtype=int)
        for i, p in enumerate(car_pos):
            g = 0
            for j in range(1, L):
                if road[(p + j) % L] >= 0:
                    break
                g += 1
            gaps[i] = g
        velocities = np.minimum(velocities, gaps)
        velocities = np.maximum(velocities, 0)

        # Rule 3: Randomisation
        rand_mask = np.random.random(len(car_pos)) < p_rand
        velocities[rand_mask] = np.maximum(velocities[rand_mask] - 1, 0)

        # Rule 4: Movement (periodic wrap)
        new_road = np.full(L, -1, dtype=int)
        new_pos  = (car_pos + velocities) % L
        new_road[new_pos] = velocities
        self.road = new_road

    def mean_flow(self):
        car_pos = np.where(self.road >= 0)[0]
        if len(car_pos) == 0:
            return 0.0
        return float(np.mean(self.road[car_pos])) * len(car_pos) / self.L


def fundamental_diagram(v_max=5, L=200, p_rand=0.3, n_densities=40,
                         T_warmup=300, T_measure=500):
    """Compute J vs rho for the periodic NaSch model."""
    densities = np.linspace(0.01, 0.99, n_densities)
    flows     = np.zeros(n_densities)
    for k, rho in enumerate(densities):
        n_cars = max(1, int(rho * L))
        sim = TrafficCA_Periodic(L=L, n_cars=n_cars, v_max=v_max, p_rand=p_rand)
        for _ in range(T_warmup):
            sim.step()
        J = 0.0
        for _ in range(T_measure):
            sim.step()
            J += sim.mean_flow()
        flows[k] = J / T_measure
    return densities, flows
