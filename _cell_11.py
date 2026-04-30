class TwoLaneNaSch:
    """Two-lane NaSch CA with CWS lane-changing.

    Parameters
    ----------
    L         : road length (cells per lane)
    n_lanes   : 1 or 2. 1 disables lane-change (used for the regression test).
    v_max     : velocity cap
    p_rand    : dawdling probability
    p_chg     : lane-change probability (applied once all four CWS
                conditions are satisfied)
    boundary  : "periodic" (closed, fixed number of cars) or "open"
                (inflow/outflow at cell 0 / cell L)
    n_cars    : required for periodic boundary; total cars across all lanes
    p_in      : inflow probability per lane per step (open boundary only)
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

        self.L = L
        self.n_lanes = n_lanes
        self.v_max = v_max
        self.p_rand = p_rand
        self.p_chg = p_chg
        self.boundary = boundary
        self.p_in = p_in

        self.road = np.full((n_lanes, L), -1, dtype=int)
        self.car_ids = np.full((n_lanes, L), -1, dtype=int)
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
            self.next_id = n_cars
        else:
            self.next_id = 0

    # ── Lane-change pre-step (CWS) ──────────────────────────────────────

    def _gap_ahead_row(self, row, positions):
        """For each pos, distance to next car ahead in `row` (periodic/open)."""
        L = self.L
        gaps = np.empty(len(positions), dtype=int)
        if self.boundary == "periodic":
            for i, p in enumerate(positions):
                g = 0
                for j in range(1, L):
                    if row[(p + j) % L] >= 0:
                        break
                    g += 1
                gaps[i] = g
        else:
            for i, p in enumerate(positions):
                g = 0
                hit = False
                for j in range(p + 1, L):
                    if row[j] >= 0:
                        hit = True
                        break
                    g += 1
                if not hit:
                    g = L - p
                gaps[i] = g
        return gaps

    def _gap_behind_row(self, row, positions):
        """For each pos, distance to nearest car *behind* in `row`."""
        L = self.L
        gaps = np.empty(len(positions), dtype=int)
        if self.boundary == "periodic":
            for i, p in enumerate(positions):
                g = 0
                for j in range(1, L):
                    if row[(p - j) % L] >= 0:
                        break
                    g += 1
                gaps[i] = g
        else:
            for i, p in enumerate(positions):
                g = 0
                hit = False
                for j in range(p - 1, -1, -1):
                    if row[j] >= 0:
                        hit = True
                        break
                    g += 1
                if not hit:
                    g = p  # free run back to cell 0
                gaps[i] = g
        return gaps

    def _lane_change_substep(self):
        """Apply CWS symmetric lane-change rule in parallel."""
        new_road = self.road.copy()
        new_ids = self.car_ids.copy()

        for src in (0, 1):
            dst = 1 - src
            car_cols = np.where(self.road[src] >= 0)[0]
            if len(car_cols) == 0:
                continue

            vels = self.road[src, car_cols]
            gap_own = self._gap_ahead_row(self.road[src], car_cols)
            gap_other_ahead = self._gap_ahead_row(self.road[dst], car_cols)
            gap_other_behind = self._gap_behind_row(self.road[dst], car_cols)
            adj_empty = self.road[dst, car_cols] < 0

            can = (
                (gap_own < vels + 1)
                & (gap_other_ahead > gap_own)
                & adj_empty
                & (gap_other_behind > self.v_max)
            )
            roll = np.random.random(len(car_cols)) < self.p_chg
            switch = can & roll

            switch_cols = car_cols[switch]
            new_road[src, switch_cols] = -1
            new_ids[src, switch_cols] = -1
            new_road[dst, switch_cols] = vels[switch]
            new_ids[dst, switch_cols] = self.car_ids[src, switch_cols]

        self.road = new_road
        self.car_ids = new_ids

    # ── Longitudinal update (per lane, same 4-rule recipe as single-lane) ──

    def _red_light_positions(self):
        """Base class: no lights. Subclass overrides."""
        return np.array([], dtype=int)

    def _step_lane_periodic(self, lane, red_cols):
        row = self.road[lane]
        ids = self.car_ids[lane]
        L = self.L
        v_max = self.v_max
        p_rand = self.p_rand

        car_pos = np.where(row >= 0)[0]
        if len(car_pos) == 0:
            return

        # Obstacles in this lane = cars + red lights (lights span both lanes)
        if len(red_cols):
            obs_mask = row >= 0
            obs_mask = obs_mask.copy()
            obs_mask[red_cols] = True
        else:
            obs_mask = row >= 0

        velocities = row[car_pos].copy()
        velocities = np.minimum(velocities + 1, v_max)

        gaps = np.empty(len(car_pos), dtype=int)
        for i, p in enumerate(car_pos):
            g = 0
            for j in range(1, L):
                if obs_mask[(p + j) % L]:
                    break
                g += 1
            gaps[i] = g
        velocities = np.minimum(velocities, gaps)
        velocities = np.maximum(velocities, 0)

        rand_mask = np.random.random(len(car_pos)) < p_rand
        velocities[rand_mask] = np.maximum(velocities[rand_mask] - 1, 0)

        new_row = np.full(L, -1, dtype=int)
        new_ids = np.full(L, -1, dtype=int)
        new_pos = (car_pos + velocities) % L
        new_row[new_pos] = velocities
        new_ids[new_pos] = ids[car_pos]

        self.road[lane] = new_row
        self.car_ids[lane] = new_ids

    def _step_lane_open(self, lane, red_cols, record_flow):
        row = self.road[lane]
        ids = self.car_ids[lane]
        L = self.L
        v_max = self.v_max
        p_rand = self.p_rand

        car_pos = np.where(row >= 0)[0]
        if len(car_pos) == 0:
            return

        if len(red_cols):
            obstacles = np.sort(np.concatenate([car_pos, red_cols]))
        else:
            obstacles = car_pos

        velocities = row[car_pos].copy()
        velocities = np.minimum(velocities + 1, v_max)

        gaps = np.full(len(car_pos), L, dtype=int)
        for i, p in enumerate(car_pos):
            idx = np.searchsorted(obstacles, p, side="right")
            if idx < len(obstacles):
                gaps[i] = obstacles[idx] - p - 1
            else:
                gaps[i] = L - p
        velocities = np.minimum(velocities, gaps)
        velocities = np.maximum(velocities, 0)

        rand_mask = np.random.random(len(car_pos)) < p_rand
        velocities[rand_mask] = np.maximum(velocities[rand_mask] - 1, 0)

        new_row = np.full(L, -1, dtype=int)
        new_ids = np.full(L, -1, dtype=int)
        new_pos = car_pos + velocities
        exited = new_pos >= L
        if record_flow:
            self.outflow_count += int(np.sum(exited))
        staying = ~exited
        new_row[new_pos[staying]] = velocities[staying]
        new_ids[new_pos[staying]] = ids[car_pos][staying]

        self.road[lane] = new_row
        self.car_ids[lane] = new_ids

    def _inflow(self):
        for lane in range(self.n_lanes):
            if self.road[lane, 0] < 0 and np.random.random() < self.p_in:
                self.road[lane, 0] = 0
                self.car_ids[lane, 0] = self.next_id
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