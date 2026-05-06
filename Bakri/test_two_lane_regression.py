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

    if test_homog_backcompat() != 0:
        return 1
    if test_race_condition() != 0:
        return 1
    if test_heterogeneity_fd() != 0:
        return 1
    if test_lane_symmetry() != 0:
        return 1
    if test_velocity_sanity() != 0:
        return 1

    return 0


# ─── Heterogeneity tests (Phase 1) ──────────────────────────────────────

def test_homog_backcompat():
    """Explicit v_max_slow=v_max, v_max_fast=v_max, f_slow=0 must be
    bit-identical to the implicit-default homogeneous run."""
    print("\n--- T_HOMOG_BACKCOMPAT ---")
    V_MAX, L, P_RAND, N_DENS, T_W, T_M = 5, 200, 0.3, 20, 300, 500

    np.random.seed(42)
    rho_a, J_a = two_lane_fd(V_MAX, L, P_RAND, N_DENS, T_W, T_M)

    # Explicit homogeneous via heterogeneity API.
    np.random.seed(42)
    densities = np.linspace(0.01, 0.99, N_DENS)
    flows = np.zeros(N_DENS)
    for k, rho in enumerate(densities):
        n_cars = max(1, int(rho * L))
        sim = TwoLaneNaSch(
            L=L, n_lanes=1, v_max=V_MAX, p_rand=P_RAND, p_chg=0.0,
            boundary="periodic", n_cars=n_cars,
            v_max_slow=V_MAX, v_max_fast=V_MAX, f_slow=0.0,
        )
        for _ in range(T_W):
            sim.step()
        J = 0.0
        for _ in range(T_M):
            sim.step()
            J += sim.mean_flow()
        flows[k] = J / T_M

    max_diff = float(np.max(np.abs(flows - J_a)))
    print(f"max |J_explicit_homog - J_default| = {max_diff:.3e}")
    if max_diff < 1e-12:
        print("PASS (bit-identical)")
        return 0
    print("FAIL — explicit homogeneous diverged from default")
    return 1


def test_race_condition():
    """Two cars at (0,10) and (1,10), each with the cell across mutually
    occupied. Snapshot semantics must make condition (iii) fail for both,
    so zero swaps occur regardless of how the lane-change rolls."""
    print("\n--- T_RACE_CONDITION ---")
    swaps = 0
    L = 20
    for seed in range(1000):
        np.random.seed(seed)
        sim = TwoLaneNaSch(
            L=L, n_lanes=2, v_max=5, p_rand=0.0, p_chg=1.0,
            boundary="periodic", n_cars=2,
        )
        # Force-place cars at (0, 10) and (1, 10), velocity 5.
        sim.road[:] = -1
        sim.car_ids[:] = -1
        sim.v_max_arr[:] = -1
        sim.road[0, 10] = 5
        sim.road[1, 10] = 5
        sim.car_ids[0, 10] = 0
        sim.car_ids[1, 10] = 1
        sim.v_max_arr[0, 10] = 5
        sim.v_max_arr[1, 10] = 5
        sim._lane_change_substep()
        if sim.road[0, 10] < 0 or sim.road[1, 10] < 0:
            swaps += 1
    print(f"{swaps} swaps over 1000 trials (expected 0)")
    if swaps == 0:
        print("PASS")
        return 0
    print("FAIL — snapshot semantics violated")
    return 1


def _periodic_J(L, v_max, p_rand, p_chg, n_cars, T_W, T_M, seed,
                v_max_slow=None, v_max_fast=None, f_slow=0.0):
    np.random.seed(seed)
    sim = TwoLaneNaSch(
        L=L, n_lanes=2, v_max=v_max, p_rand=p_rand, p_chg=p_chg,
        boundary="periodic", n_cars=n_cars,
        v_max_slow=v_max_slow, v_max_fast=v_max_fast, f_slow=f_slow,
    )
    for _ in range(T_W):
        sim.step()
    J = 0.0
    for _ in range(T_M):
        sim.step()
        J += sim.mean_flow()
    return J / T_M


def test_heterogeneity_fd():
    """At fixed density, J(f_slow=0) ≈ homogeneous v_max=v_max_fast,
    J(f_slow=1) ≈ homogeneous v_max=v_max_slow, and J(f_slow=0.5) lies
    strictly between."""
    print("\n--- T_HETEROGENEITY_FD ---")
    L, p_rand, p_chg = 200, 0.3, 0.3
    rho = 0.3
    n_cars = int(rho * 2 * L)
    T_W, T_M = 1000, 2000
    seeds = [0, 1, 2]

    v_fast, v_slow = 5, 2

    def avg(f_slow, vmax_arg):
        # If vmax_arg differs from v_fast, this is a homogeneous reference.
        return np.mean([
            _periodic_J(
                L=L, v_max=vmax_arg, p_rand=p_rand, p_chg=p_chg,
                n_cars=n_cars, T_W=T_W, T_M=T_M, seed=s,
                v_max_slow=v_slow, v_max_fast=v_fast,
                f_slow=f_slow,
            )
            for s in seeds
        ])

    J_f0 = avg(0.0, v_fast)        # f_slow=0 uses fast ceiling
    J_f1 = avg(1.0, v_fast)        # f_slow=1: every car drawn slow
    J_f5 = avg(0.5, v_fast)

    # Homogeneous references (no heterogeneity API at all).
    J_homog_fast = np.mean([
        _periodic_J(L=L, v_max=v_fast, p_rand=p_rand, p_chg=p_chg,
                    n_cars=n_cars, T_W=T_W, T_M=T_M, seed=s)
        for s in seeds
    ])
    J_homog_slow = np.mean([
        _periodic_J(L=L, v_max=v_slow, p_rand=p_rand, p_chg=p_chg,
                    n_cars=n_cars, T_W=T_W, T_M=T_M, seed=s)
        for s in seeds
    ])

    print(f"J(f_slow=0.0) = {J_f0:.4f}   homog v_max=5 ref = {J_homog_fast:.4f}")
    print(f"J(f_slow=0.5) = {J_f5:.4f}")
    print(f"J(f_slow=1.0) = {J_f1:.4f}   homog v_max=2 ref = {J_homog_slow:.4f}")

    ok = True
    # f_slow=0: bit-identical RNG sequence — short-circuit was applied.
    if abs(J_f0 - J_homog_fast) > 1e-12:
        print(f"FAIL — J(f_slow=0) not bit-identical to homog v_max=5: "
              f"|diff|={abs(J_f0 - J_homog_fast):.3e}")
        ok = False
    # f_slow=1: Monte-Carlo tolerance only.
    tol = 0.02
    if abs(J_f1 - J_homog_slow) > tol:
        print(f"FAIL — J(f_slow=1) far from homog v_max=2: "
              f"|diff|={abs(J_f1 - J_homog_slow):.3e} > tol={tol}")
        ok = False
    if not (J_f1 < J_f5 < J_f0):
        print("FAIL — J(0.5) does not lie strictly between J(0) and J(1)")
        ok = False
    if ok:
        print("PASS")
        return 0
    return 1


def test_lane_symmetry():
    """Open boundary, equal p_in for both lanes, with heterogeneity:
    mean per-lane density and mean per-lane velocity should agree to
    within 3·SEM over seeds."""
    print("\n--- T_LANE_SYMMETRY ---")
    L, v_max, p_rand, p_chg = 500, 5, 0.3, 0.5
    p_in, f_slow = 0.4, 0.3
    T_W, T_M = 2000, 2000
    n_seeds = 10

    rho_lane = np.zeros((n_seeds, 2))
    v_lane = np.zeros((n_seeds, 2))
    for si, seed in enumerate(range(n_seeds)):
        np.random.seed(seed)
        sim = TwoLaneNaSch(
            L=L, n_lanes=2, v_max=v_max, p_rand=p_rand, p_chg=p_chg,
            boundary="open", p_in=p_in,
            v_max_slow=2, v_max_fast=5, f_slow=f_slow,
        )
        for _ in range(T_W):
            sim.step()
        rho_acc = np.zeros(2)
        v_acc = np.zeros(2)
        n_acc = np.zeros(2)
        for _ in range(T_M):
            sim.step()
            for lane in (0, 1):
                mask = sim.road[lane] >= 0
                rho_acc[lane] += float(np.sum(mask)) / L
                if np.any(mask):
                    v_acc[lane] += float(np.sum(sim.road[lane][mask]))
                    n_acc[lane] += float(np.sum(mask))
        rho_lane[si] = rho_acc / T_M
        v_lane[si] = np.where(n_acc > 0, v_acc / np.maximum(n_acc, 1), 0.0)

    rho_diff = abs(rho_lane[:, 0].mean() - rho_lane[:, 1].mean())
    rho_sem = float(np.std(rho_lane[:, 0] - rho_lane[:, 1], ddof=1) / np.sqrt(n_seeds))
    v_diff = abs(v_lane[:, 0].mean() - v_lane[:, 1].mean())
    v_sem = float(np.std(v_lane[:, 0] - v_lane[:, 1], ddof=1) / np.sqrt(n_seeds))

    print(f"rho_lane_0 = {rho_lane[:, 0].mean():.4f}  rho_lane_1 = {rho_lane[:, 1].mean():.4f}  "
          f"|diff| = {rho_diff:.4f}  3·SEM = {3 * rho_sem:.4f}")
    print(f"v_lane_0   = {v_lane[:, 0].mean():.4f}  v_lane_1   = {v_lane[:, 1].mean():.4f}  "
          f"|diff| = {v_diff:.4f}  3·SEM = {3 * v_sem:.4f}")

    ok = (rho_diff < 3 * rho_sem) and (v_diff < 3 * v_sem)
    if ok:
        print("PASS")
        return 0
    print("FAIL — lane symmetry violated")
    return 1


def test_velocity_sanity():
    """Open boundary, mixed traffic: per-type mean velocities respect
    their individual ceilings and fast cars are clearly faster than slow.
    Uses a low p_in (free-flow regime) so fast cars can actually exploit
    their higher ceiling — a saturated road would let both types hover at
    similar speeds set by the bottleneck."""
    print("\n--- T_VELOCITY_SANITY ---")
    L, v_max, p_rand, p_chg = 500, 5, 0.3, 0.5
    v_slow, v_fast = 2, 5
    p_in, f_slow = 0.2, 0.3
    T_W, T_M = 2000, 2000

    np.random.seed(123)
    sim = TwoLaneNaSch(
        L=L, n_lanes=2, v_max=v_max, p_rand=p_rand, p_chg=p_chg,
        boundary="open", p_in=p_in,
        v_max_slow=v_slow, v_max_fast=v_fast, f_slow=f_slow,
    )
    for _ in range(T_W):
        sim.step()
    sum_v_slow = 0.0
    n_v_slow = 0
    sum_v_fast = 0.0
    n_v_fast = 0
    for _ in range(T_M):
        sim.step()
        mask = sim.road >= 0
        if np.any(mask):
            slow_mask = mask & (sim.v_max_arr == v_slow)
            fast_mask = mask & (sim.v_max_arr == v_fast)
            sum_v_slow += float(np.sum(sim.road[slow_mask]))
            n_v_slow += int(np.sum(slow_mask))
            sum_v_fast += float(np.sum(sim.road[fast_mask]))
            n_v_fast += int(np.sum(fast_mask))

    mean_v_slow = sum_v_slow / max(n_v_slow, 1)
    mean_v_fast = sum_v_fast / max(n_v_fast, 1)
    print(f"mean_v_slow = {mean_v_slow:.4f}   v_max_slow = {v_slow}")
    print(f"mean_v_fast = {mean_v_fast:.4f}")

    ok = True
    if mean_v_slow > v_slow + 1e-9:
        print("FAIL — mean_v_slow exceeds v_max_slow")
        ok = False
    if not (mean_v_fast > mean_v_slow + 0.2):
        print(f"FAIL — fast not noticeably faster (gap {mean_v_fast - mean_v_slow:.3f})")
        ok = False
    if ok:
        print("PASS")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
