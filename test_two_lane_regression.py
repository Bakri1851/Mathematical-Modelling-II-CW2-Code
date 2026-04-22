"""Regression check: two-lane NaSch with n_lanes=1, p_chg=0 must reproduce
the single-lane fundamental diagram from `notebooks/nagel_schreckenberg.ipynb`.

For identical seed and parameters the RNG draw sequence is identical, so
we expect a bit-exact match (max |J_single - J_two| == 0). The looser
Monte-Carlo tolerance is kept as a fallback in case someone later changes
the implementation in a way that perturbs RNG ordering without changing
the physics.
"""

import sys

import numpy as np

from _nasch_nb import TrafficCA_Periodic, fundamental_diagram
from nasch_two_lane import TwoLaneNaSch


def two_lane_fd(v_max, L, p_rand, n_densities, T_warmup, T_measure):
    """Same measurement protocol as `fundamental_diagram` but using TwoLaneNaSch."""
    densities = np.linspace(0.01, 0.99, n_densities)
    flows = np.zeros(n_densities)
    for k, rho in enumerate(densities):
        n_cars = max(1, int(rho * L))
        sim = TwoLaneNaSch(
            L=L,
            n_lanes=1,
            v_max=v_max,
            p_rand=p_rand,
            p_chg=0.0,
            boundary="periodic",
            n_cars=n_cars,
        )
        for _ in range(T_warmup):
            sim.step()
        J = 0.0
        for _ in range(T_measure):
            sim.step()
            J += sim.mean_flow()
        flows[k] = J / T_measure
    return densities, flows


def main():
    V_MAX = 5
    L = 200
    P_RAND = 0.3
    N_DENS = 20
    T_W = 300
    T_M = 500
    TOL = 0.02  # ~4× the Monte Carlo sigma at these params; we actually expect 0.

    # Single-lane baseline.
    np.random.seed(42)
    rho_ref, J_ref = fundamental_diagram(
        v_max=V_MAX, L=L, p_rand=P_RAND,
        n_densities=N_DENS, T_warmup=T_W, T_measure=T_M,
    )

    # Two-lane with n_lanes=1, p_chg=0 — should be identical.
    np.random.seed(42)
    rho_two, J_two = two_lane_fd(V_MAX, L, P_RAND, N_DENS, T_W, T_M)

    assert np.allclose(rho_ref, rho_two), "density grid mismatch"

    diffs = J_two - J_ref
    max_diff = float(np.max(np.abs(diffs)))

    print("  rho     J_single   J_two       diff")
    for r, j1, j2, d in zip(rho_ref, J_ref, J_two, diffs):
        print(f"  {r:5.3f}   {j1:.6f}   {j2:.6f}   {d:+.2e}")
    print(f"\nmax |J_two - J_single| = {max_diff:.3e}   tol = {TOL:.2e}")

    if max_diff < 1e-12:
        print("PASS (bit-identical — RNG traces match)")
    elif max_diff < TOL:
        print(f"PASS (within Monte-Carlo tolerance)")
    else:
        print("FAIL — two-lane with n_lanes=1, p_chg=0 does not reproduce single-lane FD")
        return 1

    # Sanity diagnostic: n_lanes=2 with lane changing on. Flow per cell should
    # remain in the same ballpark as single-lane; cars can now rearrange but
    # the density per cell is the same.
    np.random.seed(42)
    rho_test = 0.3
    sim2 = TwoLaneNaSch(
        L=L, n_lanes=2, v_max=V_MAX, p_rand=P_RAND, p_chg=1.0,
        boundary="periodic", n_cars=int(rho_test * 2 * L),
    )
    for _ in range(T_W):
        sim2.step()
    J2 = 0.0
    for _ in range(T_M):
        sim2.step()
        J2 += sim2.mean_flow()
    J2 /= T_M

    # Compare to single-lane flow at the same density for context.
    np.random.seed(42)
    sim1 = TrafficCA_Periodic(L=L, n_cars=int(rho_test * L), v_max=V_MAX, p_rand=P_RAND)
    for _ in range(T_W):
        sim1.step()
    J1 = 0.0
    for _ in range(T_M):
        sim1.step()
        J1 += sim1.mean_flow()
    J1 /= T_M
    print(f"\nSanity: at rho={rho_test}, single-lane J={J1:.4f}, "
          f"two-lane J (per cell)={J2:.4f} (p_chg=1.0)")

    # n_lanes=3 must raise.
    try:
        TwoLaneNaSch(L=10, n_lanes=3, n_cars=2)
    except ValueError:
        print("n_lanes=3 correctly rejected (ValueError)")
    else:
        print("FAIL — n_lanes=3 did not raise ValueError")
        return 1

    # p_chg=0 must produce zero lane-switch events.
    np.random.seed(42)
    sim_no_chg = TwoLaneNaSch(
        L=L, n_lanes=2, v_max=V_MAX, p_rand=P_RAND, p_chg=0.0,
        boundary="periodic", n_cars=80,
    )
    before = sim_no_chg.car_ids.copy()
    for _ in range(100):
        sim_no_chg.step()
    # With p_chg=0, a car never changes lane, so its ID stays in the same lane
    # (though its column changes). Check that every ID still present is in its
    # original lane.
    after_ids_lane0 = set(sim_no_chg.car_ids[0][sim_no_chg.car_ids[0] >= 0])
    before_ids_lane0 = set(before[0][before[0] >= 0])
    after_ids_lane1 = set(sim_no_chg.car_ids[1][sim_no_chg.car_ids[1] >= 0])
    before_ids_lane1 = set(before[1][before[1] >= 0])
    # With periodic boundary no cars enter/exit, so both sets should be equal.
    if after_ids_lane0 != before_ids_lane0 or after_ids_lane1 != before_ids_lane1:
        print("FAIL — lane-change happened with p_chg=0")
        return 1
    print("p_chg=0 confirmed: no lane changes in 100 steps")

    return 0


if __name__ == "__main__":
    sys.exit(main())
