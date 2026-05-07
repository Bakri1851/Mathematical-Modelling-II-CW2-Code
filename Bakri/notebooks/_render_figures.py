"""Render the missing report figures (F2, F3, F5, F7, F8, F9, F10, F11, F12, F13, F14).

Standalone script — does NOT depend on Jupyter. Run from `Bakri/notebooks/` with:

    python _render_figures.py

Parameters are deliberately modest so the full run finishes in a few minutes.
The publication-quality settings should bump T_MEASURE, n_seeds, and the
heatmap grid sizes; current values are noted next to each figure.
"""
import os
import sys
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure local module imports resolve regardless of cwd.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from nasch import TrafficCA, TrafficCA_Periodic, fundamental_diagram, TrafficCAWithLights
from nasch_two_lane import (
    TwoLaneNaSch,
    TwoLaneNaSchWithLights,
    two_lane_fundamental_diagram,
)
from nasch_intersection import IntersectionSystem

DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))
FIG_DIR = os.path.normpath(os.path.join(HERE, "..", "figures"))
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (8, 5),
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def load_or_compute(path, compute_fn, force=False):
    if force or not os.path.exists(path):
        result = compute_fn()
        np.savez(path, **result)
        return result
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def _timed(name, fn):
    t0 = time.time()
    fn()
    print(f"[{name}] done in {time.time() - t0:.1f}s")


# ── F2 — Open-boundary J(p_in) for several v_max ───────────────────────────
def fig_F2():
    L = 500
    P_RAND = 0.3
    T_WARMUP = 800
    T_MEASURE = 2500
    SEEDS = np.arange(3)
    P_IN_GRID = np.linspace(0.05, 0.95, 16)
    VMAX_LIST = [2, 3, 5]

    def compute():
        flow = np.zeros((len(VMAX_LIST), len(P_IN_GRID), len(SEEDS)))
        for vi, vm in enumerate(VMAX_LIST):
            for pi, p_in in enumerate(P_IN_GRID):
                for si, seed in enumerate(SEEDS):
                    np.random.seed(int(seed))
                    sim = TrafficCA(L=L, v_max=int(vm), p_rand=P_RAND, p_in=float(p_in))
                    sim.warmup(T_WARMUP)
                    flow[vi, pi, si] = sim.run(T_MEASURE)
        return dict(
            flow=flow, p_in_grid=P_IN_GRID, vmax_list=np.array(VMAX_LIST),
            L=L, p_rand=P_RAND, T_warmup=T_WARMUP, T_measure=T_MEASURE,
        )

    d = load_or_compute(os.path.join(DATA_DIR, "F2_open_J_vs_pin.npz"), compute)

    fig, ax = plt.subplots()
    for vi, vm in enumerate(d["vmax_list"]):
        m = d["flow"][vi].mean(axis=-1)
        s = d["flow"][vi].std(axis=-1)
        ax.plot(d["p_in_grid"], m, "-o", label=f"v_max = {int(vm)}")
        ax.fill_between(d["p_in_grid"], m - s, m + s, alpha=0.2)
    ax.plot([0, 1], [0, 1], "--", color="grey", label=r"J = $p_\mathrm{in}$ (insertion limit)")
    ax.set_xlabel(r"Inflow probability $p_\mathrm{in}$")
    ax.set_ylabel("Throughput J [cars / timestep]")
    ax.set_title("F2 — Single-lane open-boundary throughput vs inflow")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(FIG_DIR, "01_open.png"))
    plt.close(fig)


# ── F3 — Space-time diagram (single-lane, stop-and-go waves) ───────────────
def fig_F3():
    L = 200
    V_MAX = 5
    P_RAND = 0.3
    P_IN = 0.5
    T_WARMUP = 500
    T_RECORD = 250

    np.random.seed(0)
    sim = TrafficCA(L=L, v_max=V_MAX, p_rand=P_RAND, p_in=P_IN)
    sim.warmup(T_WARMUP)
    grid = np.zeros((T_RECORD, L), dtype=int)
    for t in range(T_RECORD):
        grid[t] = sim.get_snapshot()
        sim.step(record_flow=False, record_travel_times=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(grid, aspect="auto", cmap="binary", interpolation="nearest")
    ax.set_xlabel("Position [cell]")
    ax.set_ylabel("Time [timestep]")
    ax.set_title(f"F3 — Single-lane space–time diagram (L={L}, $v_{{max}}$={V_MAX}, $p_{{rand}}$={P_RAND}, $p_{{in}}$={P_IN})")
    ax.invert_yaxis()
    ax.grid(False)
    fig.savefig(os.path.join(FIG_DIR, "F3_space_time.png"))
    plt.close(fig)


# ── F7 — Two-lane fundamental diagram for several p_chg + single-lane ref ──
def fig_F7():
    L = 200
    V_MAX = 5
    P_RAND = 0.3
    N_DENS = 25
    T_W = 500
    T_M = 1500
    PCHG_LIST = [0.0, 0.25, 0.5, 1.0]

    def compute():
        out = {"p_chg_list": np.array(PCHG_LIST), "L": L, "v_max": V_MAX,
               "p_rand": P_RAND, "T_warmup": T_W, "T_measure": T_M}
        # Single-lane reference
        np.random.seed(0)
        rho_ref, J_ref = fundamental_diagram(
            v_max=V_MAX, L=L, p_rand=P_RAND,
            n_densities=N_DENS, T_warmup=T_W, T_measure=T_M,
        )
        out["rho_single"] = rho_ref
        out["J_single"] = J_ref
        # Two-lane curves
        all_J = np.zeros((len(PCHG_LIST), N_DENS))
        for ci, pc in enumerate(PCHG_LIST):
            np.random.seed(0)
            rho2, J2 = two_lane_fundamental_diagram(
                v_max=V_MAX, L=L, p_rand=P_RAND, n_lanes=2,
                p_chg=float(pc), n_densities=N_DENS,
                T_warmup=T_W, T_measure=T_M,
            )
            all_J[ci] = J2
        out["rho_two"] = rho2
        out["J_two"] = all_J
        return out

    d = load_or_compute(os.path.join(DATA_DIR, "F7_two_lane_fd.npz"), compute)

    fig, ax = plt.subplots()
    ax.plot(d["rho_single"], d["J_single"], "--", color="grey",
            label="single-lane reference")
    for ci, pc in enumerate(d["p_chg_list"]):
        ax.plot(d["rho_two"], d["J_two"][ci], "-o", markersize=4,
                label=f"two-lane, $p_{{chg}}$ = {float(pc)}")
    ax.set_xlabel(r"Density $\rho$ [cars / cell]")
    ax.set_ylabel("Flow J [cars / timestep / cell]")
    ax.set_title(f"F7 — Two-lane fundamental diagram (L={L}, $v_{{max}}$={V_MAX}, $p_{{rand}}$={P_RAND})")
    ax.legend()
    fig.savefig(os.path.join(FIG_DIR, "F7_two_lane_fd.png"))
    plt.close(fig)


# ── F9 — Intersection symmetric η-sweep (validation flagship) ──────────────
def fig_F9():
    L = 200
    V_MAX = 5
    P_RAND = 0.3
    P_IN = 0.4
    T = 60
    T_WARMUP = 500
    T_MEASURE = 1500
    SEEDS = np.arange(3)
    ETA_GRID = np.linspace(0.1, 0.9, 17)

    def compute():
        J1 = np.zeros((len(ETA_GRID), len(SEEDS)))
        J2 = np.zeros((len(ETA_GRID), len(SEEDS)))
        for ei, eta in enumerate(ETA_GRID):
            for si, seed in enumerate(SEEDS):
                np.random.seed(int(seed))
                sys_ = IntersectionSystem(
                    L=L, v_max=V_MAX, p_rand=P_RAND,
                    p_in_1=P_IN, p_in_2=P_IN, T=T, eta=float(eta),
                    seed=int(seed),
                )
                sys_.warmup(T_WARMUP)
                j1, j2, _ = sys_.run(T_MEASURE)
                J1[ei, si] = j1
                J2[ei, si] = j2
        return dict(
            J1=J1, J2=J2, eta_grid=ETA_GRID, seeds=SEEDS,
            L=L, v_max=V_MAX, p_rand=P_RAND, p_in=P_IN, T=T,
            T_warmup=T_WARMUP, T_measure=T_MEASURE,
        )

    d = load_or_compute(os.path.join(DATA_DIR, "F9_intersection_eta.npz"), compute)
    J1m = d["J1"].mean(-1); J1s = d["J1"].std(-1)
    J2m = d["J2"].mean(-1); J2s = d["J2"].std(-1)
    Jt = J1m + J2m

    fig, ax = plt.subplots()
    ax.errorbar(d["eta_grid"], J1m, yerr=J1s, fmt="-o", markersize=4, label=r"$J_1$ (road 1)")
    ax.errorbar(d["eta_grid"], J2m, yerr=J2s, fmt="-s", markersize=4, label=r"$J_2$ (road 2)")
    ax.plot(d["eta_grid"], Jt, "-^", markersize=4, label=r"$J_{total}$")
    ax.axvline(0.5, linestyle="--", color="grey", alpha=0.6,
               label=r"$\eta = 0.5$ (symmetry axis)")
    ax.set_xlabel(r"Green-time split $\eta = T_{g,1} / T$")
    ax.set_ylabel("Throughput J [cars / timestep]")
    ax.set_title(f"F9 — Intersection η-sweep, symmetric demand ($p_{{in}}={P_IN}$, T={T})")
    ax.legend()
    fig.savefig(os.path.join(FIG_DIR, "03_symmetric.png"))
    plt.close(fig)

    # Print symmetry-error summary for the validation paragraph.
    sym_err = float(np.max(np.abs(Jt - Jt[::-1])))
    seed_std = float(np.max(np.maximum(J1s, J2s)))
    print(f"  symmetry max|J_t(eta) - J_t(1-eta)| = {sym_err:.4f}, "
          f"max seed-sigma = {seed_std:.4f}")


# ── F10 — Intersection asymmetric demand heatmap, η=0.5 ────────────────────
def fig_F10():
    L = 200
    V_MAX = 5
    P_RAND = 0.3
    T = 60
    ETA = 0.5
    T_WARMUP = 500
    T_MEASURE = 1500
    SEEDS = np.arange(2)
    PIN_GRID = np.linspace(0.1, 0.7, 7)

    def compute():
        Jt = np.zeros((len(PIN_GRID), len(PIN_GRID), len(SEEDS)))
        for i, pin1 in enumerate(PIN_GRID):
            for j, pin2 in enumerate(PIN_GRID):
                for si, seed in enumerate(SEEDS):
                    np.random.seed(int(seed))
                    sys_ = IntersectionSystem(
                        L=L, v_max=V_MAX, p_rand=P_RAND,
                        p_in_1=float(pin1), p_in_2=float(pin2),
                        T=T, eta=ETA, seed=int(seed),
                    )
                    sys_.warmup(T_WARMUP)
                    _, _, jt = sys_.run(T_MEASURE)
                    Jt[i, j, si] = jt
        return dict(
            Jt=Jt, pin_grid=PIN_GRID, seeds=SEEDS,
            eta=ETA, T=T, L=L, v_max=V_MAX, p_rand=P_RAND,
            T_warmup=T_WARMUP, T_measure=T_MEASURE,
        )

    d = load_or_compute(os.path.join(DATA_DIR, "F10_intersection_asym.npz"), compute)
    mean_Jt = d["Jt"].mean(-1)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mean_Jt.T, origin="lower", cmap="viridis",
                   extent=[d["pin_grid"][0], d["pin_grid"][-1],
                           d["pin_grid"][0], d["pin_grid"][-1]],
                   aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$J_{total}$ [cars / timestep]")
    ax.set_xlabel(r"Inflow road 1, $p_{in,1}$")
    ax.set_ylabel(r"Inflow road 2, $p_{in,2}$")
    ax.set_title(f"F10 — Intersection $J_{{total}}$ heatmap, η={ETA}, T={d['T']}")
    fig.savefig(os.path.join(FIG_DIR, "03_asym_heatmap.png"))
    plt.close(fig)


# ── F5 — J(φ) at N=2 (two-lane lights phase-offset sweep) ──────────────────
def fig_F5():
    L = 500
    V_MAX = 5
    P_RAND = 0.3
    P_IN = 0.4
    P_CHG = 0.3
    N = 2
    T_CYCLE = 60
    T_GREEN = 30
    T_WARMUP = 800
    T_MEASURE = 2000
    SEEDS = np.arange(2)
    PHI_GRID = np.arange(0, T_CYCLE, 5)  # 12 points

    def compute():
        flow = np.zeros((len(PHI_GRID), len(SEEDS)))
        for pi, phi in enumerate(PHI_GRID):
            phase_offsets = np.array([0, int(phi)], dtype=int)
            for si, seed in enumerate(SEEDS):
                np.random.seed(int(seed))
                sim = TwoLaneNaSchWithLights(
                    L=L, n_lanes=2, v_max=V_MAX, p_rand=P_RAND,
                    p_chg=P_CHG, boundary="open", p_in=P_IN,
                    N=N, T_cycle=T_CYCLE, T_green=T_GREEN,
                    phase_offsets=phase_offsets,
                )
                sim.warmup(T_WARMUP)
                flow[pi, si] = sim.run(T_MEASURE)
        return dict(
            flow=flow, phi_grid=PHI_GRID, seeds=SEEDS,
            L=L, v_max=V_MAX, p_rand=P_RAND, p_in=P_IN, p_chg=P_CHG,
            N=N, T_cycle=T_CYCLE, T_green=T_GREEN,
            T_warmup=T_WARMUP, T_measure=T_MEASURE,
        )

    d = load_or_compute(os.path.join(DATA_DIR, "F5_phi_sweep.npz"), compute)
    m = d["flow"].mean(-1); s = d["flow"].std(-1)

    # Reference offsets
    v_free = V_MAX - P_RAND
    spacing = L / (N + 1)
    # Correct green-wave: phi_i = -i*spacing/v_free mod T_cycle
    phi_greenwave = int(round((-spacing / v_free) % T_CYCLE))
    phi_anti = T_CYCLE // 2

    fig, ax = plt.subplots()
    ax.plot(d["phi_grid"], m, "-o")
    ax.fill_between(d["phi_grid"], m - s, m + s, alpha=0.2)
    ax.axvline(0, linestyle="--", color="tab:green", alpha=0.7,
               label=r"in-phase ($\varphi=0$)")
    ax.axvline(phi_greenwave, linestyle="--", color="tab:red", alpha=0.7,
               label=fr"green-wave ($\varphi$ = {phi_greenwave})")
    ax.axvline(phi_anti, linestyle="--", color="tab:purple", alpha=0.7,
               label=fr"anti-phase ($\varphi$ = {phi_anti})")
    ax.set_xlabel(r"Phase offset $\varphi$ of light 2 [timesteps]")
    ax.set_ylabel("Throughput J [cars / timestep]")
    ax.set_title(f"F5 — Two-lane J vs phase offset (N={N}, T_cycle={T_CYCLE})")
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(FIG_DIR, "02_phi_sweep.png"))
    plt.close(fig)


# ── F8 — (φ, p_chg) heatmap ────────────────────────────────────────────────
def fig_F8():
    L = 500
    V_MAX = 5
    P_RAND = 0.3
    P_IN = 0.4
    N = 2
    T_CYCLE = 60
    T_GREEN = 30
    T_WARMUP = 600
    T_MEASURE = 1500
    SEEDS = np.arange(2)
    PHI_GRID = np.array([0, 10, 20, 30, 40, 50])
    PCHG_GRID = np.array([0.0, 0.25, 0.5, 1.0])

    def compute():
        flow = np.zeros((len(PHI_GRID), len(PCHG_GRID), len(SEEDS)))
        for i, phi in enumerate(PHI_GRID):
            phase_offsets = np.array([0, int(phi)], dtype=int)
            for j, pc in enumerate(PCHG_GRID):
                for si, seed in enumerate(SEEDS):
                    np.random.seed(int(seed))
                    sim = TwoLaneNaSchWithLights(
                        L=L, n_lanes=2, v_max=V_MAX, p_rand=P_RAND,
                        p_chg=float(pc), boundary="open", p_in=P_IN,
                        N=N, T_cycle=T_CYCLE, T_green=T_GREEN,
                        phase_offsets=phase_offsets,
                    )
                    sim.warmup(T_WARMUP)
                    flow[i, j, si] = sim.run(T_MEASURE)
        return dict(
            flow=flow, phi_grid=PHI_GRID, pchg_grid=PCHG_GRID, seeds=SEEDS,
            L=L, v_max=V_MAX, p_rand=P_RAND, p_in=P_IN, N=N,
            T_cycle=T_CYCLE, T_green=T_GREEN,
            T_warmup=T_WARMUP, T_measure=T_MEASURE,
        )

    d = load_or_compute(os.path.join(DATA_DIR, "F8_phi_pchg_heatmap.npz"), compute)
    mean_flow = d["flow"].mean(-1)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(mean_flow.T, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(np.arange(len(d["phi_grid"])))
    ax.set_xticklabels([str(int(x)) for x in d["phi_grid"]])
    ax.set_yticks(np.arange(len(d["pchg_grid"])))
    ax.set_yticklabels([f"{float(x):.2f}" for x in d["pchg_grid"]])
    ax.set_xlabel(r"Phase offset $\varphi$ [timesteps]")
    ax.set_ylabel(r"Lane-change probability $p_{chg}$")
    ax.set_title(f"F8 — Two-lane J heatmap, $p_{{in}}$={d['p_in']}, N={d['N']}, T_cycle={d['T_cycle']}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("J [cars / timestep]")
    # Annotate cells with values
    for i in range(len(d["phi_grid"])):
        for j in range(len(d["pchg_grid"])):
            ax.text(i, j, f"{mean_flow[i, j]:.2f}", ha="center", va="center",
                    color="white", fontsize=10)
    fig.savefig(os.path.join(FIG_DIR, "02_heatmap.png"))
    plt.close(fig)


# ── F14 — Capstone: in-phase vs green-wave space-time juxtaposition ────────
def fig_F14():
    L = 400
    V_MAX = 5
    P_RAND = 0.3
    P_IN = 0.4
    N = 2
    T_CYCLE = 60
    T_GREEN = 30
    T_WARMUP = 400
    T_RECORD = 200

    def record(phase_offsets):
        np.random.seed(0)
        sim = TrafficCAWithLights(
            L=L, v_max=V_MAX, p_rand=P_RAND, p_in=P_IN,
            N=N, T_cycle=T_CYCLE, T_green=T_GREEN,
            phase_offsets=phase_offsets,
        )
        for _ in range(T_WARMUP):
            sim.step()
        grid = np.zeros((T_RECORD, L), dtype=int)
        for t in range(T_RECORD):
            grid[t] = sim.get_snapshot()
            sim.step()
        return grid

    v_free = V_MAX - P_RAND
    spacing = L / (N + 1)
    # Correct green-wave: phi_i = -i*spacing/v_free mod T_cycle (light JUST turns green as car arrives)
    phi_greenwave = int(round((-spacing / v_free) % T_CYCLE))

    grid_inphase = record(np.array([0, 0], dtype=int))
    grid_wave = record(np.array([0, phi_greenwave], dtype=int))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, grid, title in [
        (axes[0], grid_inphase, f"In-phase ($\\varphi$ = 0)"),
        (axes[1], grid_wave, f"Green-wave ($\\varphi$ = {phi_greenwave})"),
    ]:
        ax.imshow(grid, aspect="auto", cmap="binary", interpolation="nearest")
        # Mark light positions as horizontal red lines at x = light_pos
        for li in range(N):
            light_x = L * (li + 1) // (N + 1)
            ax.axvline(light_x, color="red", alpha=0.35, linewidth=0.7)
        ax.set_xlabel("Position [cell]")
        ax.set_title(title)
        ax.invert_yaxis()
        ax.grid(False)
    axes[0].set_ylabel("Time [timestep]")
    fig.suptitle(
        f"F14 — Single-lane space–time, in-phase vs green-wave (N={N}, T_cycle={T_CYCLE})",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "F14_capstone_spacetime.png"))
    plt.close(fig)


# ── F11/F12/F13 — Sensitivity tornado, response curves, joint heatmap ──────
def fig_F11_F12_F13():
    """OAT sensitivity around a fixed two-lane-with-lights baseline.

    Tornado plot ranks |S_i| where S_i = (J(1.1p) - J(0.9p)) / (0.2 * J_base).
    Integer-valued parameters (N, T_cycle, T_green, v_max) use ±1-step
    perturbations; the perturbation symmetry-fraction is reported in the
    caption rather than enforced uniformly.
    """
    # Baseline.
    L = 400
    V_MAX = 5
    P_RAND = 0.3
    P_IN = 0.4
    P_CHG = 0.3
    N = 2
    T_CYCLE = 60
    T_GREEN = 30
    PHI = 30  # baseline phase offset for light 2 (anti-phase as a non-extreme starting point)
    T_WARMUP = 600
    T_MEASURE = 2000
    SEEDS = np.arange(3)

    def run_one(p_rand, p_in, p_chg, v_max, N_, T_cycle, T_green, phi):
        flow = np.zeros(len(SEEDS))
        for si, seed in enumerate(SEEDS):
            np.random.seed(int(seed))
            phase_offsets = np.array([0, int(phi)], dtype=int) if N_ == 2 else np.zeros(N_, dtype=int)
            if N_ != 2:
                # Re-derive a sensible phase offset for the green-wave at the baseline v_free.
                v_free = max(0.1, v_max - p_rand)
                spacing = L / (N_ + 1)
                phase_offsets = np.array([
                    int(round((i * spacing / v_free) % T_cycle)) for i in range(N_)
                ], dtype=int)
            sim = TwoLaneNaSchWithLights(
                L=L, n_lanes=2, v_max=int(v_max), p_rand=float(p_rand),
                p_chg=float(p_chg), boundary="open", p_in=float(p_in),
                N=int(N_), T_cycle=int(T_cycle), T_green=int(T_green),
                phase_offsets=phase_offsets,
            )
            sim.warmup(T_WARMUP)
            flow[si] = sim.run(T_MEASURE)
        return flow.mean(), flow.std()

    def compute():
        # Baseline.
        J_base, _ = run_one(P_RAND, P_IN, P_CHG, V_MAX, N, T_CYCLE, T_GREEN, PHI)

        # Each parameter has a (low, high) perturbation pair.
        # Continuous params: ±10%. Integer params: ±1 step (with sane lower bounds).
        params = [
            ("p_rand",  P_RAND * 0.9,  P_RAND * 1.1),
            ("p_in",    P_IN   * 0.9,  P_IN   * 1.1),
            ("p_chg",   P_CHG  * 0.9,  P_CHG  * 1.1),
            ("v_max",   max(2, V_MAX - 1), V_MAX + 1),
            ("N",       max(1, N - 1),     N + 1),
            ("T_cycle", T_CYCLE - 6,   T_CYCLE + 6),     # ±10%
            ("T_green", T_GREEN - 3,   T_GREEN + 3),     # ±10%
            ("phi", max(0, PHI - 6), PHI + 6),     # ±10% of T_cycle
        ]
        sens_low = np.zeros(len(params))
        sens_high = np.zeros(len(params))
        for i, (name, lo, hi) in enumerate(params):
            if name == "p_rand":
                Jl, _ = run_one(lo, P_IN, P_CHG, V_MAX, N, T_CYCLE, T_GREEN, PHI)
                Jh, _ = run_one(hi, P_IN, P_CHG, V_MAX, N, T_CYCLE, T_GREEN, PHI)
            elif name == "p_in":
                Jl, _ = run_one(P_RAND, lo, P_CHG, V_MAX, N, T_CYCLE, T_GREEN, PHI)
                Jh, _ = run_one(P_RAND, hi, P_CHG, V_MAX, N, T_CYCLE, T_GREEN, PHI)
            elif name == "p_chg":
                Jl, _ = run_one(P_RAND, P_IN, lo, V_MAX, N, T_CYCLE, T_GREEN, PHI)
                Jh, _ = run_one(P_RAND, P_IN, hi, V_MAX, N, T_CYCLE, T_GREEN, PHI)
            elif name == "v_max":
                Jl, _ = run_one(P_RAND, P_IN, P_CHG, lo, N, T_CYCLE, T_GREEN, PHI)
                Jh, _ = run_one(P_RAND, P_IN, P_CHG, hi, N, T_CYCLE, T_GREEN, PHI)
            elif name == "N":
                Jl, _ = run_one(P_RAND, P_IN, P_CHG, V_MAX, lo, T_CYCLE, T_GREEN, PHI)
                Jh, _ = run_one(P_RAND, P_IN, P_CHG, V_MAX, hi, T_CYCLE, T_GREEN, PHI)
            elif name == "T_cycle":
                # Keep T_green at the baseline GREEN_FRAC = 0.5 of the perturbed cycle.
                Jl, _ = run_one(P_RAND, P_IN, P_CHG, V_MAX, N, lo, max(1, lo // 2), PHI)
                Jh, _ = run_one(P_RAND, P_IN, P_CHG, V_MAX, N, hi, max(1, hi // 2), PHI)
            elif name == "T_green":
                Jl, _ = run_one(P_RAND, P_IN, P_CHG, V_MAX, N, T_CYCLE, lo, PHI)
                Jh, _ = run_one(P_RAND, P_IN, P_CHG, V_MAX, N, T_CYCLE, hi, PHI)
            else:  # phi
                Jl, _ = run_one(P_RAND, P_IN, P_CHG, V_MAX, N, T_CYCLE, T_GREEN, int(lo))
                Jh, _ = run_one(P_RAND, P_IN, P_CHG, V_MAX, N, T_CYCLE, T_GREEN, int(hi))
            sens_low[i] = Jl
            sens_high[i] = Jh

        # Sensitivity coefficient: scale into normalised central-difference.
        S = (sens_high - sens_low) / (0.2 * J_base)
        names = np.array([p[0] for p in params])
        return dict(
            J_base=np.array(J_base), sens_low=sens_low, sens_high=sens_high,
            S=S, names=names,
            L=L, v_max=V_MAX, p_rand=P_RAND, p_in=P_IN, p_chg=P_CHG,
            N=N, T_cycle=T_CYCLE, T_green=T_GREEN, phi=PHI,
            T_warmup=T_WARMUP, T_measure=T_MEASURE,
        )

    d = load_or_compute(os.path.join(DATA_DIR, "F11_tornado.npz"), compute)
    S = d["S"]
    names = d["names"]
    order = np.argsort(np.abs(S))[::-1]

    # F11 — Tornado plot.
    fig, ax = plt.subplots(figsize=(8, 5))
    ypos = np.arange(len(order))
    bars = ax.barh(ypos, S[order],
                   color=["tab:blue" if S[i] > 0 else "tab:red" for i in order])
    ax.set_yticks(ypos)
    ax.set_yticklabels([names[i] for i in order])
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Sensitivity coefficient $S$")
    ax.set_title(
        f"F11 — Tornado plot (baseline J = {float(d['J_base']):.3f}; "
        "blue: amplifying, red: attenuating)"
    )
    fig.savefig(os.path.join(FIG_DIR, "F11_tornado.png"))
    plt.close(fig)

    # F12 — top-3 response curves at finer resolution.
    top3 = order[:3]
    response_path = os.path.join(DATA_DIR, "F12_top3_response.npz")

    def compute_response():
        L_ = L
        SEEDS_local = np.arange(3)
        T_W_, T_M_ = 500, 1500
        # Param-specific sweep grids around the baseline.
        sweeps = {}
        for idx in top3:
            name = names[idx]
            if name == "p_rand":
                grid = np.linspace(0.05, 0.6, 8)
                fn = lambda v: run_one(v, P_IN, P_CHG, V_MAX, N, T_CYCLE, T_GREEN, PHI)
            elif name == "p_in":
                grid = np.linspace(0.1, 0.9, 9)
                fn = lambda v: run_one(P_RAND, v, P_CHG, V_MAX, N, T_CYCLE, T_GREEN, PHI)
            elif name == "p_chg":
                grid = np.linspace(0.0, 1.0, 11)
                fn = lambda v: run_one(P_RAND, P_IN, v, V_MAX, N, T_CYCLE, T_GREEN, PHI)
            elif name == "v_max":
                grid = np.array([2, 3, 4, 5, 6, 7])
                fn = lambda v: run_one(P_RAND, P_IN, P_CHG, int(v), N, T_CYCLE, T_GREEN, PHI)
            elif name == "N":
                grid = np.array([1, 2, 3, 4, 5])
                fn = lambda v: run_one(P_RAND, P_IN, P_CHG, V_MAX, int(v), T_CYCLE, T_GREEN, PHI)
            elif name == "T_cycle":
                grid = np.array([20, 40, 60, 80, 120])
                fn = lambda v: run_one(P_RAND, P_IN, P_CHG, V_MAX, N, int(v), int(v) // 2, PHI)
            elif name == "T_green":
                grid = np.array([10, 20, 30, 40, 50])
                fn = lambda v: run_one(P_RAND, P_IN, P_CHG, V_MAX, N, T_CYCLE, int(v), PHI)
            else:  # phi
                grid = np.arange(0, T_CYCLE, 6)
                fn = lambda v: run_one(P_RAND, P_IN, P_CHG, V_MAX, N, T_CYCLE, T_GREEN, int(v))
            J_grid = np.zeros(len(grid))
            for k, val in enumerate(grid):
                J_grid[k], _ = fn(val)
            sweeps[name] = (np.asarray(grid, dtype=float), J_grid)
        # Pickle-style npz: flatten to per-parameter arrays.
        out = {"order_top3": np.array([names[i] for i in top3])}
        for name, (g, j) in sweeps.items():
            out[f"grid_{name}"] = g
            out[f"flow_{name}"] = j
        return out

    r = load_or_compute(response_path, compute_response)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for k, idx in enumerate(top3):
        name = names[idx]
        # The npz keys may differ from the original dict (varphi is stored as r"$\varphi$" too).
        g_key = f"grid_{name}"
        j_key = f"flow_{name}"
        # The dict was saved with the literal name we passed in compute_response,
        # which uses the same names array entry — so this lookup is safe.
        if g_key not in r:
            continue
        grid = r[g_key]
        Jvals = r[j_key]
        axes[k].plot(grid, Jvals, "-o")
        axes[k].axhline(float(d["J_base"]), linestyle="--", color="grey", alpha=0.7,
                        label="baseline J")
        axes[k].set_xlabel(name)
        axes[k].set_ylabel("J [cars / timestep]")
        axes[k].set_title(f"{name}  (S = {S[idx]:+.2f})")
        axes[k].legend()
    fig.suptitle("F12 — Top-3 sensitivity response curves")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "F12_top3_response.png"))
    plt.close(fig)

    # F13 — joint sweep heatmap (T_cycle × T_green).
    F13_path = os.path.join(DATA_DIR, "F13_Tcycle_Tgreen.npz")
    TCYCLE_GRID = np.array([20, 40, 60, 80, 100])
    TGREEN_FRAC_GRID = np.array([0.2, 0.4, 0.5, 0.6, 0.8])

    def compute_joint():
        flow = np.zeros((len(TCYCLE_GRID), len(TGREEN_FRAC_GRID), len(SEEDS)))
        for i, Tc in enumerate(TCYCLE_GRID):
            for j, gf in enumerate(TGREEN_FRAC_GRID):
                Tg = max(1, int(round(gf * Tc)))
                for si, seed in enumerate(SEEDS):
                    np.random.seed(int(seed))
                    phase_offsets = np.array([0, PHI], dtype=int)
                    sim = TwoLaneNaSchWithLights(
                        L=L, n_lanes=2, v_max=V_MAX, p_rand=P_RAND,
                        p_chg=P_CHG, boundary="open", p_in=P_IN,
                        N=N, T_cycle=int(Tc), T_green=Tg,
                        phase_offsets=phase_offsets,
                    )
                    sim.warmup(T_WARMUP)
                    flow[i, j, si] = sim.run(T_MEASURE)
        return dict(flow=flow, Tcycle_grid=TCYCLE_GRID, Tgreen_frac_grid=TGREEN_FRAC_GRID)

    j13 = load_or_compute(F13_path, compute_joint)
    mean_flow_13 = j13["flow"].mean(-1)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(mean_flow_13.T, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(np.arange(len(j13["Tcycle_grid"])))
    ax.set_xticklabels([str(int(x)) for x in j13["Tcycle_grid"]])
    ax.set_yticks(np.arange(len(j13["Tgreen_frac_grid"])))
    ax.set_yticklabels([f"{float(x):.1f}" for x in j13["Tgreen_frac_grid"]])
    ax.set_xlabel(r"$T_{cycle}$ [timesteps]")
    ax.set_ylabel(r"Green fraction $T_{green} / T_{cycle}$")
    ax.set_title(f"F13 — Joint sweep heatmap, $p_{{in}}$={P_IN}, $p_{{chg}}$={P_CHG}, N={N}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("J [cars / timestep]")
    for i in range(len(j13["Tcycle_grid"])):
        for j in range(len(j13["Tgreen_frac_grid"])):
            ax.text(i, j, f"{mean_flow_13[i, j]:.2f}", ha="center", va="center",
                    color="white", fontsize=10)
    fig.savefig(os.path.join(FIG_DIR, "F13_Tcycle_Tgreen.png"))
    plt.close(fig)


# ── Single-lane lights helpers (used by F4_single, F5_single, F6_single, F_strats) ──
def _greenwave_offsets(N, T_cycle, v_max, p_rand, L):
    """Phase offsets that align consecutive lights at the free-flow speed.

    A car leaving light 0 at the moment it turns green travels at v_free for
    `spacing/v_free` steps before reaching light 1; light 1 should JUST turn
    green at that arrival time. With phase_i(t) = (t + φ_i) mod T_cycle and
    light green iff phase < T_green, "just turning green" means phase = 0.
    Setting (travel_time + φ_1) mod T_cycle = 0 gives φ_i = (-i × travel_time) mod T_cycle.
    The opposite sign (positive) is sub-optimal: light 1 turns green BEFORE
    the car arrives, wasting green time.
    """
    v_free = max(0.1, v_max - p_rand)
    spacing = L / (N + 1)
    return np.array([
        int(round((-i * spacing / v_free) % T_cycle)) for i in range(N)
    ], dtype=int)


def _measure_J_single_lights(L, v_max, p_rand, p_in, N, T_cycle, T_green,
                              phase_offsets, T_warmup, T_measure, seed):
    """Run TrafficCAWithLights for one (param-set, seed) and return through-flow J."""
    np.random.seed(int(seed))
    sim = TrafficCAWithLights(
        L=L, v_max=int(v_max), p_rand=float(p_rand), p_in=float(p_in),
        N=int(N), T_cycle=int(T_cycle), T_green=int(T_green),
        phase_offsets=np.asarray(phase_offsets, dtype=int),
    )
    sim.warmup(T_warmup)
    return sim.run(T_measure)


# ── F4_single — single-lane J(T_cycle) for several N ──────────────────────
def fig_F4_single():
    L = 500
    V_MAX = 5
    P_RAND = 0.3
    P_IN = 0.4
    SEEDS = np.arange(3)
    TC_GRID = np.array([10, 20, 30, 40, 60, 80, 100, 150])
    N_GRID = np.array([1, 2, 3])
    T_WARMUP = 800
    T_MEASURE = 2500

    def compute():
        flow = np.zeros((len(N_GRID), len(TC_GRID), len(SEEDS)))
        for ni, N in enumerate(N_GRID):
            for ti, Tc in enumerate(TC_GRID):
                T_green = int(Tc) // 2
                phi = _greenwave_offsets(int(N), int(Tc), V_MAX, P_RAND, L)
                for si, seed in enumerate(SEEDS):
                    flow[ni, ti, si] = _measure_J_single_lights(
                        L, V_MAX, P_RAND, P_IN, int(N), int(Tc), T_green,
                        phi, T_WARMUP, T_MEASURE, int(seed),
                    )
        return dict(flow=flow, T_cycle_grid=TC_GRID, N_grid=N_GRID, seeds=SEEDS,
                    L=L, v_max=V_MAX, p_rand=P_RAND, p_in=P_IN,
                    T_warmup=T_WARMUP, T_measure=T_MEASURE)

    d = load_or_compute(os.path.join(DATA_DIR, "F4_single_J_vs_Tcycle.npz"), compute)

    fig, ax = plt.subplots()
    v_free = V_MAX - P_RAND
    for ni, N in enumerate(d["N_grid"]):
        m = d["flow"][ni].mean(-1)
        s = d["flow"][ni].std(-1)
        ax.plot(d["T_cycle_grid"], m, "-o", label=f"N = {int(N)}")
        ax.fill_between(d["T_cycle_grid"], m - s, m + s, alpha=0.2)
        # Resonance marker for this N: T_cycle = 2 * d / v_free where d = L/(N+1)
        Tc_res = 2 * (L / (int(N) + 1)) / v_free
        ax.axvline(Tc_res, linestyle="--", alpha=0.4,
                   color=ax.lines[-1].get_color())
    ax.set_xlabel(r"Cycle length $T_\mathrm{cycle}$ [timesteps]")
    ax.set_ylabel("Throughput J [cars / timestep]")
    ax.set_title(f"F4 — Single-lane J vs $T_\\mathrm{{cycle}}$ (green-wave; $p_\\mathrm{{in}}$={P_IN})")
    ax.legend()
    fig.savefig(os.path.join(FIG_DIR, "F4_single_J_vs_Tcycle.png"))
    plt.close(fig)


# ── F5_single — single-lane J(phi) at N=2 ────────────────────────────────
def fig_F5_single():
    L = 500
    V_MAX = 5
    P_RAND = 0.3
    P_IN = 0.4
    N = 2
    T_CYCLE = 60
    T_GREEN = 30
    SEEDS = np.arange(3)
    PHI_GRID = np.arange(0, T_CYCLE, 5)  # 12 points
    T_WARMUP = 800
    T_MEASURE = 2500

    def compute():
        flow = np.zeros((len(PHI_GRID), len(SEEDS)))
        for pi, phi in enumerate(PHI_GRID):
            phase_offsets = np.array([0, int(phi)], dtype=int)
            for si, seed in enumerate(SEEDS):
                flow[pi, si] = _measure_J_single_lights(
                    L, V_MAX, P_RAND, P_IN, N, T_CYCLE, T_GREEN,
                    phase_offsets, T_WARMUP, T_MEASURE, int(seed),
                )
        return dict(flow=flow, phi_grid=PHI_GRID, seeds=SEEDS,
                    L=L, v_max=V_MAX, p_rand=P_RAND, p_in=P_IN,
                    N=N, T_cycle=T_CYCLE, T_green=T_GREEN,
                    T_warmup=T_WARMUP, T_measure=T_MEASURE)

    d = load_or_compute(os.path.join(DATA_DIR, "F5_single_phi_sweep.npz"), compute)
    m = d["flow"].mean(-1)
    s = d["flow"].std(-1)

    v_free = V_MAX - P_RAND
    spacing = L / (N + 1)
    phi_greenwave = int(round((spacing / v_free) % T_CYCLE))
    phi_anti = T_CYCLE // 2

    fig, ax = plt.subplots()
    ax.plot(d["phi_grid"], m, "-o")
    ax.fill_between(d["phi_grid"], m - s, m + s, alpha=0.2)
    ax.axvline(0, linestyle="--", color="tab:green", alpha=0.7,
               label=r"in-phase ($\varphi$ = 0)")
    ax.axvline(phi_greenwave, linestyle="--", color="tab:red", alpha=0.7,
               label=fr"green-wave ($\varphi$ = {phi_greenwave})")
    ax.axvline(phi_anti, linestyle="--", color="tab:purple", alpha=0.7,
               label=fr"anti-phase ($\varphi$ = {phi_anti})")
    ax.set_xlabel(r"Phase offset $\varphi$ of light 2 [timesteps]")
    ax.set_ylabel("Throughput J [cars / timestep]")
    ax.set_title(f"F5 — Single-lane J vs phase offset (N={N}, $T_\\mathrm{{cycle}}$={T_CYCLE})")
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(FIG_DIR, "F5_single_phi_sweep.png"))
    plt.close(fig)


# ── F6_single — single-lane J(N) for sync vs green-wave ──────────────────
def fig_F6_single():
    L = 500
    V_MAX = 5
    P_RAND = 0.3
    P_IN = 0.4
    T_CYCLE = 60
    T_GREEN = 30
    SEEDS = np.arange(3)
    N_GRID = np.array([1, 2, 3, 4, 5, 6, 8, 10])
    STRATEGIES = ["sync", "green-wave"]
    T_WARMUP = 800
    T_MEASURE = 2500

    def compute():
        flow = np.zeros((len(N_GRID), len(STRATEGIES), len(SEEDS)))
        for ni, N in enumerate(N_GRID):
            sync_phi = np.zeros(int(N), dtype=int)
            wave_phi = _greenwave_offsets(int(N), T_CYCLE, V_MAX, P_RAND, L)
            for si_strat, strat_phi in enumerate([sync_phi, wave_phi]):
                for si, seed in enumerate(SEEDS):
                    flow[ni, si_strat, si] = _measure_J_single_lights(
                        L, V_MAX, P_RAND, P_IN, int(N), T_CYCLE, T_GREEN,
                        strat_phi, T_WARMUP, T_MEASURE, int(seed),
                    )
        return dict(flow=flow, N_grid=N_GRID, strategies=np.array(STRATEGIES),
                    seeds=SEEDS, L=L, v_max=V_MAX, p_rand=P_RAND, p_in=P_IN,
                    T_cycle=T_CYCLE, T_green=T_GREEN,
                    T_warmup=T_WARMUP, T_measure=T_MEASURE)

    d = load_or_compute(os.path.join(DATA_DIR, "F6_single_J_vs_N.npz"), compute)

    fig, ax = plt.subplots()
    for si_strat, strat in enumerate(STRATEGIES):
        m = d["flow"][:, si_strat, :].mean(-1)
        s = d["flow"][:, si_strat, :].std(-1)
        ax.plot(d["N_grid"], m, "-o", label=strat)
        ax.fill_between(d["N_grid"], m - s, m + s, alpha=0.2)
    ax.set_xlabel("Number of lights N")
    ax.set_ylabel("Throughput J [cars / timestep]")
    ax.set_title(f"F6 — Single-lane J vs N (sync vs green-wave; $T_\\mathrm{{cycle}}$={T_CYCLE})")
    ax.legend()
    fig.savefig(os.path.join(FIG_DIR, "F6_single_J_vs_N.png"))
    plt.close(fig)


# ── F_strats — single-lane J(N) for four coordination strategies ──────────
def fig_F_strats():
    L = 500
    V_MAX = 5
    P_RAND = 0.3
    P_IN = 0.4
    T_CYCLE = 60
    T_GREEN = 30
    SEEDS = np.arange(3)
    N_GRID = np.array([2, 3, 4, 5, 6, 8])
    STRATEGIES = ["sync", "green-wave", "anti-greenwave", "random"]
    N_RANDOM_DRAWS = 5
    T_WARMUP = 800
    T_MEASURE = 2500

    def offsets_for(N, strategy, rng_seed=None):
        v_free = max(0.1, V_MAX - P_RAND)
        spacing = L / (int(N) + 1)
        if strategy == "sync":
            return np.zeros(int(N), dtype=int)
        if strategy == "green-wave":
            # Correct: light_i turns green AS car arrives (φ_i = -i·spacing/v_free).
            return np.array([
                int(round((-i * spacing / v_free) % T_CYCLE)) for i in range(int(N))
            ], dtype=int)
        if strategy == "anti-greenwave":
            # Reverse: light_i turns green AS car arrives going BACKWARDS — so as
            # car (going forward) arrives the light has been red the whole window.
            return np.array([
                int(round((i * spacing / v_free) % T_CYCLE)) for i in range(int(N))
            ], dtype=int)
        if strategy == "random":
            rng = np.random.RandomState(int(rng_seed))
            return rng.randint(0, T_CYCLE, size=int(N))
        raise ValueError(f"unknown strategy: {strategy}")

    def compute():
        # For sync / green-wave / anti-greenwave: 1 offset config × n_seeds seeds.
        # For random: N_RANDOM_DRAWS independent offset configs, each over n_seeds.
        flow = np.zeros((len(N_GRID), len(STRATEGIES), len(SEEDS)))
        flow_random_draws = np.zeros((len(N_GRID), N_RANDOM_DRAWS, len(SEEDS)))
        for ni, N in enumerate(N_GRID):
            for si_strat, strat in enumerate(STRATEGIES):
                if strat == "random":
                    for di in range(N_RANDOM_DRAWS):
                        phi = offsets_for(int(N), "random", rng_seed=1000 * int(N) + di)
                        for si, seed in enumerate(SEEDS):
                            flow_random_draws[ni, di, si] = _measure_J_single_lights(
                                L, V_MAX, P_RAND, P_IN, int(N), T_CYCLE, T_GREEN,
                                phi, T_WARMUP, T_MEASURE, int(seed),
                            )
                    flow[ni, si_strat, :] = flow_random_draws[ni].mean(axis=0)  # avg over draws
                else:
                    phi = offsets_for(int(N), strat)
                    for si, seed in enumerate(SEEDS):
                        flow[ni, si_strat, si] = _measure_J_single_lights(
                            L, V_MAX, P_RAND, P_IN, int(N), T_CYCLE, T_GREEN,
                            phi, T_WARMUP, T_MEASURE, int(seed),
                        )
        return dict(flow=flow, flow_random_draws=flow_random_draws,
                    N_grid=N_GRID, strategies=np.array(STRATEGIES),
                    seeds=SEEDS, L=L, v_max=V_MAX, p_rand=P_RAND, p_in=P_IN,
                    T_cycle=T_CYCLE, T_green=T_GREEN, n_random_draws=N_RANDOM_DRAWS,
                    T_warmup=T_WARMUP, T_measure=T_MEASURE)

    d = load_or_compute(os.path.join(DATA_DIR, "F_strats.npz"), compute)

    fig, ax = plt.subplots()
    colors = {"sync": "tab:blue", "green-wave": "tab:green",
              "anti-greenwave": "tab:red", "random": "tab:gray"}
    for si_strat, strat in enumerate(STRATEGIES):
        m = d["flow"][:, si_strat, :].mean(-1)
        s = d["flow"][:, si_strat, :].std(-1)
        ax.plot(d["N_grid"], m, "-o", label=strat, color=colors[strat])
        ax.fill_between(d["N_grid"], m - s, m + s, alpha=0.2, color=colors[strat])
    ax.set_xlabel("Number of lights N")
    ax.set_ylabel("Throughput J [cars / timestep]")
    ax.set_title(f"F-strats — Single-lane J vs N for four coordination strategies")
    ax.legend()
    fig.savefig(os.path.join(FIG_DIR, "F_strats.png"))
    plt.close(fig)

    # Sanity-check ordering: print mean J at N=4 for each strategy.
    idx_n4 = int(np.argmin(np.abs(d["N_grid"] - 4)))
    print(f"  F-strats at N={int(d['N_grid'][idx_n4])}:")
    for si_strat, strat in enumerate(STRATEGIES):
        Jm = float(d["flow"][idx_n4, si_strat, :].mean())
        print(f"    {strat:<16} J = {Jm:.4f}")


def main():
    t_total = time.time()
    _timed("F2 (single-lane open J(p_in))", fig_F2)
    _timed("F3 (single-lane space-time)", fig_F3)
    _timed("F7 (two-lane fundamental diagram)", fig_F7)
    _timed("F9 (intersection symmetric η-sweep)", fig_F9)
    _timed("F10 (intersection asymmetric heatmap)", fig_F10)
    _timed("F5 (two-lane phi sweep)", fig_F5)
    _timed("F8 (two-lane phi×p_chg heatmap)", fig_F8)
    _timed("F14 (capstone in-phase vs green-wave)", fig_F14)
    _timed("F11+F12+F13 (sensitivity)", fig_F11_F12_F13)
    _timed("F4_single (J vs T_cycle, single-lane)", fig_F4_single)
    _timed("F5_single (J vs phi, single-lane)", fig_F5_single)
    _timed("F6_single (J vs N, sync vs green-wave)", fig_F6_single)
    _timed("F_strats (J vs N, 4 strategies)", fig_F_strats)
    print(f"\nALL DONE in {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
