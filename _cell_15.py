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
