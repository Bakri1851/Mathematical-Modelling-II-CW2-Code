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

    def reset(self):
        self.road[:]    = -1
        self.car_ids[:] = -1
        self.next_id    = 0
        self.time       = 0
        self.outflow_count = 0

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
        gaps = np.full(len(positions), self.L, dtype=int)
        if len(obstacle_positions) == 0:
            # no obstacles — gap is distance to road end
            gaps = self.L - positions
            return gaps
        for i, p in enumerate(positions):
            # find first obstacle strictly ahead
            idx = np.searchsorted(obstacle_positions, p, side='right')
            if idx < len(obstacle_positions):
                gaps[i] = obstacle_positions[idx] - p - 1
            else:
                gaps[i] = self.L - p  # free run to exit
        return gaps

    # ── Red-light obstacle positions (overridden by subclass) ─────────────────

    def _red_light_positions(self):
        """Return array of cell indices where a red light blocks traffic."""
        return np.array([], dtype=int)

    # ── Main update step ──────────────────────────────────────────────────────

    def step(self, record_flow=False):
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
            self.next_id   += 1

    # ── Simulation runners ────────────────────────────────────────────────────

    def warmup(self, T=None):
        T = T or T_WARMUP
        for _ in range(T):
            self.step(record_flow=False)
        self.outflow_count = 0

    def run(self, T=None):
        """Run T measurement steps, return mean flow (cars/timestep)."""
        T = T or T_MEASURE
        self.outflow_count = 0
        for _ in range(T):
            self.step(record_flow=True)
        return self.outflow_count / T

    def get_snapshot(self):
        """Return binary occupancy array (1=car, 0=empty)."""
        return (self.road >= 0).astype(int)

    def density(self):
        return np.sum(self.road >= 0) / self.L


print("TrafficCA class defined.")