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
