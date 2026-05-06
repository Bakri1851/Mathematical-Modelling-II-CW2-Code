"""Build 04_two_lane_sensitivity.ipynb programmatically.

Writes a fresh notebook with parameter constants in cell 1, autoreload,
load_or_compute caching, and one section per sweep. Outputs are NOT
embedded; run the notebook with `jupyter nbconvert --to notebook --execute`
to fill outputs.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# 04 — Two-lane sensitivity (homogeneous, p_chg = 0.3)\n\n"
    "Two-lane parity of `nagel_sensitivity.ipynb`. All sweeps use "
    "homogeneous traffic (`f_slow = 0`) and the standard two-lane "
    "open-boundary protocol with traffic lights. Heterogeneity sweeps "
    "live in 05.\n\n"
    "Phase-offset (φ) sensitivity is in **02**; we don't reproduce it. "
    "Mixed-traffic results are in **05**.\n\n"
    "**Cache convention** — every `.npz` preserves the full per-seed "
    "shape `(n_..., n_seeds)` so SD bands are recomputed at plot time, "
    "not at save time."
))

cells.append(nbf.v4.new_code_cell(
    "%load_ext autoreload\n"
    "%autoreload 2\n"
    "\n"
    "import os, time, numpy as np, matplotlib.pyplot as plt\n"
    "from tqdm.auto import tqdm\n"
    "\n"
    "from nasch_two_lane import TwoLaneNaSch, TwoLaneNaSchWithLights\n"
    "\n"
    "plt.rcParams.update({\n"
    "    'figure.figsize': (8, 5),\n"
    "    'font.size': 12,\n"
    "    'axes.grid': True,\n"
    "    'grid.alpha': 0.3,\n"
    "    'axes.labelsize': 12,\n"
    "    'axes.titlesize': 13,\n"
    "    'legend.fontsize': 11,\n"
    "    'savefig.dpi': 150,\n"
    "    'savefig.bbox': 'tight',\n"
    "})\n"
    "\n"
    "DATA_DIR = '../data'\n"
    "FIGURES_DIR = '../figures'\n"
    "os.makedirs(DATA_DIR, exist_ok=True)\n"
    "os.makedirs(FIGURES_DIR, exist_ok=True)\n"
    "\n"
    "# ─── Global parameters ────────────────────────────────────────────\n"
    "L            = 500\n"
    "V_MAX        = 5\n"
    "P_RAND       = 0.3\n"
    "P_IN         = 0.4\n"
    "T_WARMUP     = 3000\n"
    "T_MEASURE    = 6000\n"
    "N_SEEDS      = 3\n"
    "PCHG_DEFAULT = 0.3\n"
    "SEEDS        = np.arange(N_SEEDS)\n"
    "V_FREE       = V_MAX - P_RAND   # canonical free-flow speed\n"
    "\n"
    "# Cache convention: per-seed shape preserved so SD bands can be\n"
    "# regenerated without resimulating.\n"
    "\n"
    "def load_or_compute(path, compute_fn, force=False):\n"
    "    \"\"\"Load `path` if it exists; else compute, save, and return.\"\"\"\n"
    "    if force or not os.path.exists(path):\n"
    "        result = compute_fn()\n"
    "        np.savez(path, **result)\n"
    "        return result\n"
    "    with np.load(path) as f:\n"
    "        return {k: f[k] for k in f.files}\n"
    "\n"
    "def greenwave_offsets(N, T_cycle, v_free=V_FREE, L_road=L):\n"
    "    \"\"\"Phase offsets that align consecutive lights at v_free.\"\"\"\n"
    "    s = L_road / (N + 1)\n"
    "    return np.array([int(round((i * s / v_free) % T_cycle))\n"
    "                     for i in range(N)])\n"
    "\n"
    "def measure_J(sim, T_warmup, T_measure):\n"
    "    \"\"\"Open-boundary throughput: outflow per timestep across both lanes.\"\"\"\n"
    "    sim.warmup(T_warmup)\n"
    "    return sim.run(T_measure)\n"
    "\n"
    "print(f'L={L}, v_max={V_MAX}, p_rand={P_RAND}, p_in={P_IN}')\n"
    "print(f'T_warmup={T_WARMUP}, T_measure={T_MEASURE}, n_seeds={N_SEEDS}')\n"
    "print(f'v_free (canonical) = {V_FREE}')\n"
))

# ── Section 2 ──
cells.append(nbf.v4.new_markdown_cell(
    "## §2 — Throughput vs N (number of lights)\n\n"
    "Sweep `N ∈ {1, 2, 3, 4, 5, 6, 8, 10}` for two coordination strategies: "
    "**sync** (all lights at phase 0) and **offset** (greenwave at v_free). "
    "Fixed `T_cycle = 60`, `T_green = 30`."
))

cells.append(nbf.v4.new_code_cell(
    "N_GRID = np.array([1, 2, 3, 4, 5, 6, 8, 10])\n"
    "STRATEGIES = ['sync', 'offset']\n"
    "T_CYCLE_S2 = 60\n"
    "T_GREEN_S2 = 30\n"
    "\n"
    "def compute_J_vs_N():\n"
    "    flow = np.zeros((len(N_GRID), len(STRATEGIES), N_SEEDS))\n"
    "    t0 = time.time()\n"
    "    for ni, N in enumerate(tqdm(N_GRID, desc='N')):\n"
    "        for si, strategy in enumerate(STRATEGIES):\n"
    "            phi = (np.zeros(N, dtype=int) if strategy == 'sync'\n"
    "                   else greenwave_offsets(N, T_CYCLE_S2))\n"
    "            for ki, seed in enumerate(SEEDS):\n"
    "                np.random.seed(seed)\n"
    "                sim = TwoLaneNaSchWithLights(\n"
    "                    L=L, n_lanes=2, v_max=V_MAX, p_rand=P_RAND,\n"
    "                    p_chg=PCHG_DEFAULT, boundary='open', p_in=P_IN,\n"
    "                    N=int(N), T_cycle=T_CYCLE_S2, T_green=T_GREEN_S2,\n"
    "                    phase_offsets=phi,\n"
    "                )\n"
    "                flow[ni, si, ki] = measure_J(sim, T_WARMUP, T_MEASURE)\n"
    "    return dict(flow=flow, N_grid=N_GRID, strategies=np.array(STRATEGIES),\n"
    "                seeds=SEEDS, T_cycle=T_CYCLE_S2, T_green=T_GREEN_S2,\n"
    "                T_warmup=T_WARMUP, T_measure=T_MEASURE,\n"
    "                wall_clock=time.time() - t0)\n"
    "\n"
    "S2 = load_or_compute(f'{DATA_DIR}/04_J_vs_N.npz', compute_J_vs_N)\n"
    "print(f'wall-clock: {float(S2[\"wall_clock\"]):.1f} s')\n"
))

cells.append(nbf.v4.new_code_cell(
    "fig, ax = plt.subplots()\n"
    "for si, strat in enumerate(STRATEGIES):\n"
    "    m = S2['flow'][:, si, :].mean(axis=-1)\n"
    "    s = S2['flow'][:, si, :].std(axis=-1)\n"
    "    ax.plot(S2['N_grid'], m, '-o', label=strat)\n"
    "    ax.fill_between(S2['N_grid'], m - s, m + s, alpha=0.2)\n"
    "ax.set_xlabel('Number of lights N')\n"
    "ax.set_ylabel('Throughput J (cars / timestep)')\n"
    "ax.set_title('§2 — J vs N')\n"
    "ax.legend()\n"
    "fig.savefig(f'{FIGURES_DIR}/04_J_vs_N.png')\n"
    "plt.show()\n"
))

cells.append(nbf.v4.new_markdown_cell(
    "**Figure 04.2 — Throughput vs number of lights.** Two-lane open boundary "
    f"with `L={500}, v_max=5, p_rand=0.3, p_in=0.4, p_chg=0.3, "
    "T_cycle=60, T_green=30, n_seeds=3, T_warmup=3000, T_measure=6000`. "
    "Solid curves are mean over seeds; shaded bands are ±1 SD. "
    "Sync = all lights in phase; offset = greenwave at v_free = 4.7."
))

# ── Section 3 ──
cells.append(nbf.v4.new_markdown_cell(
    "## §3 — Throughput vs T_cycle\n\n"
    "Sweep `T_cycle ∈ {10, 20, 30, 40, 60, 80, 100, 150}`. Fixed `N=2`, "
    "green fraction = 0.5, greenwave offset. Vertical dashed line marks "
    "the theoretical resonance `T_cycle = 2·d/v_free` where d = L/(N+1)."
))

cells.append(nbf.v4.new_code_cell(
    "TC_GRID = np.array([10, 20, 30, 40, 60, 80, 100, 150])\n"
    "N_S3 = 2\n"
    "\n"
    "def compute_J_vs_Tcycle():\n"
    "    flow = np.zeros((len(TC_GRID), N_SEEDS))\n"
    "    t0 = time.time()\n"
    "    for ti, Tc in enumerate(tqdm(TC_GRID, desc='Tc')):\n"
    "        T_green = int(Tc) // 2\n"
    "        phi = greenwave_offsets(N_S3, int(Tc))\n"
    "        for ki, seed in enumerate(SEEDS):\n"
    "            np.random.seed(seed)\n"
    "            sim = TwoLaneNaSchWithLights(\n"
    "                L=L, n_lanes=2, v_max=V_MAX, p_rand=P_RAND,\n"
    "                p_chg=PCHG_DEFAULT, boundary='open', p_in=P_IN,\n"
    "                N=N_S3, T_cycle=int(Tc), T_green=T_green,\n"
    "                phase_offsets=phi,\n"
    "            )\n"
    "            flow[ti, ki] = measure_J(sim, T_WARMUP, T_MEASURE)\n"
    "    return dict(flow=flow, T_cycle_grid=TC_GRID, seeds=SEEDS,\n"
    "                N=N_S3, T_warmup=T_WARMUP, T_measure=T_MEASURE,\n"
    "                wall_clock=time.time() - t0)\n"
    "\n"
    "S3 = load_or_compute(f'{DATA_DIR}/04_J_vs_Tcycle.npz', compute_J_vs_Tcycle)\n"
    "print(f'wall-clock: {float(S3[\"wall_clock\"]):.1f} s')\n"
))

cells.append(nbf.v4.new_code_cell(
    "fig, ax = plt.subplots()\n"
    "m = S3['flow'].mean(axis=-1)\n"
    "s = S3['flow'].std(axis=-1)\n"
    "ax.plot(S3['T_cycle_grid'], m, '-o')\n"
    "ax.fill_between(S3['T_cycle_grid'], m - s, m + s, alpha=0.2)\n"
    "d = L / (N_S3 + 1)\n"
    "Tc_resonance = 2 * d / V_FREE\n"
    "ax.axvline(Tc_resonance, color='k', linestyle='--', alpha=0.5,\n"
    "           label=f'2·d/v_free = {Tc_resonance:.0f}')\n"
    "ax.set_xlabel('T_cycle (timesteps)')\n"
    "ax.set_ylabel('Throughput J (cars / timestep)')\n"
    "ax.set_title('§3 — J vs T_cycle (greenwave, N=2)')\n"
    "ax.legend()\n"
    "fig.savefig(f'{FIGURES_DIR}/04_J_vs_Tcycle.png')\n"
    "plt.show()\n"
))

# ── Section 4 ──
cells.append(nbf.v4.new_markdown_cell(
    "## §4 — Throughput vs green fraction\n\n"
    "Sweep `T_green / T_cycle ∈ {0.1, ..., 0.9}` at fixed `N=2`, "
    "`T_cycle=60`, greenwave offset."
))

cells.append(nbf.v4.new_code_cell(
    "GF_GRID = np.arange(0.1, 0.95, 0.1)\n"
    "T_CYCLE_S4 = 60\n"
    "N_S4 = 2\n"
    "\n"
    "def compute_J_vs_green():\n"
    "    flow = np.zeros((len(GF_GRID), N_SEEDS))\n"
    "    t0 = time.time()\n"
    "    phi = greenwave_offsets(N_S4, T_CYCLE_S4)\n"
    "    for gi, gf in enumerate(tqdm(GF_GRID, desc='gf')):\n"
    "        T_green = max(1, int(round(gf * T_CYCLE_S4)))\n"
    "        for ki, seed in enumerate(SEEDS):\n"
    "            np.random.seed(seed)\n"
    "            sim = TwoLaneNaSchWithLights(\n"
    "                L=L, n_lanes=2, v_max=V_MAX, p_rand=P_RAND,\n"
    "                p_chg=PCHG_DEFAULT, boundary='open', p_in=P_IN,\n"
    "                N=N_S4, T_cycle=T_CYCLE_S4, T_green=T_green,\n"
    "                phase_offsets=phi,\n"
    "            )\n"
    "            flow[gi, ki] = measure_J(sim, T_WARMUP, T_MEASURE)\n"
    "    return dict(flow=flow, green_frac_grid=GF_GRID, seeds=SEEDS,\n"
    "                N=N_S4, T_cycle=T_CYCLE_S4,\n"
    "                T_warmup=T_WARMUP, T_measure=T_MEASURE,\n"
    "                wall_clock=time.time() - t0)\n"
    "\n"
    "S4 = load_or_compute(f'{DATA_DIR}/04_J_vs_green.npz', compute_J_vs_green)\n"
    "print(f'wall-clock: {float(S4[\"wall_clock\"]):.1f} s')\n"
))

cells.append(nbf.v4.new_code_cell(
    "fig, ax = plt.subplots()\n"
    "m = S4['flow'].mean(axis=-1)\n"
    "s = S4['flow'].std(axis=-1)\n"
    "ax.plot(S4['green_frac_grid'], m, '-o')\n"
    "ax.fill_between(S4['green_frac_grid'], m - s, m + s, alpha=0.2)\n"
    "ax.set_xlabel('Green fraction T_green / T_cycle')\n"
    "ax.set_ylabel('Throughput J (cars / timestep)')\n"
    "ax.set_title('§4 — J vs green fraction (N=2, T_cycle=60)')\n"
    "fig.savefig(f'{FIGURES_DIR}/04_J_vs_green.png')\n"
    "plt.show()\n"
))

# ── Section 5 ──
cells.append(nbf.v4.new_markdown_cell(
    "## §5 — Throughput vs v_max\n\n"
    "Sweep `v_max ∈ {1, 2, 3, 5, 7, 10}` at fixed `N=2, T_cycle=60, "
    "T_green=30`, greenwave offset."
))

cells.append(nbf.v4.new_code_cell(
    "VMAX_GRID = np.array([1, 2, 3, 5, 7, 10])\n"
    "T_CYCLE_S5 = 60\n"
    "T_GREEN_S5 = 30\n"
    "N_S5 = 2\n"
    "\n"
    "def compute_J_vs_vmax():\n"
    "    flow = np.zeros((len(VMAX_GRID), N_SEEDS))\n"
    "    t0 = time.time()\n"
    "    for vi, vm in enumerate(tqdm(VMAX_GRID, desc='vmax')):\n"
    "        v_free_local = max(0.1, vm - P_RAND)\n"
    "        phi = np.array([int(round((i * (L / (N_S5 + 1)) / v_free_local) % T_CYCLE_S5))\n"
    "                        for i in range(N_S5)])\n"
    "        for ki, seed in enumerate(SEEDS):\n"
    "            np.random.seed(seed)\n"
    "            sim = TwoLaneNaSchWithLights(\n"
    "                L=L, n_lanes=2, v_max=int(vm), p_rand=P_RAND,\n"
    "                p_chg=PCHG_DEFAULT, boundary='open', p_in=P_IN,\n"
    "                N=N_S5, T_cycle=T_CYCLE_S5, T_green=T_GREEN_S5,\n"
    "                phase_offsets=phi,\n"
    "            )\n"
    "            flow[vi, ki] = measure_J(sim, T_WARMUP, T_MEASURE)\n"
    "    return dict(flow=flow, vmax_grid=VMAX_GRID, seeds=SEEDS,\n"
    "                T_warmup=T_WARMUP, T_measure=T_MEASURE,\n"
    "                wall_clock=time.time() - t0)\n"
    "\n"
    "S5 = load_or_compute(f'{DATA_DIR}/04_J_vs_vmax.npz', compute_J_vs_vmax)\n"
    "print(f'wall-clock: {float(S5[\"wall_clock\"]):.1f} s')\n"
))

cells.append(nbf.v4.new_code_cell(
    "fig, ax = plt.subplots()\n"
    "m = S5['flow'].mean(axis=-1)\n"
    "s = S5['flow'].std(axis=-1)\n"
    "ax.plot(S5['vmax_grid'], m, '-o')\n"
    "ax.fill_between(S5['vmax_grid'], m - s, m + s, alpha=0.2)\n"
    "ax.set_xlabel('v_max')\n"
    "ax.set_ylabel('Throughput J (cars / timestep)')\n"
    "ax.set_title('§5 — J vs v_max (N=2, T_cycle=60, greenwave)')\n"
    "fig.savefig(f'{FIGURES_DIR}/04_J_vs_vmax.png')\n"
    "plt.show()\n"
))

# ── Section 6 ──
cells.append(nbf.v4.new_markdown_cell(
    "## §6 — (N, T_cycle) optimisation heatmap\n\n"
    "2D sweep `N × T_cycle ∈ {1..5} × {10, 20, 40, 60, 100}`. Fixed green "
    "fraction = 0.5, greenwave offset. Mean J reported with empirical "
    "argmax marked."
))

cells.append(nbf.v4.new_code_cell(
    "N_GRID_S6 = np.array([1, 2, 3, 4, 5])\n"
    "TC_GRID_S6 = np.array([10, 20, 40, 60, 100])\n"
    "\n"
    "def compute_heatmap():\n"
    "    flow = np.zeros((len(N_GRID_S6), len(TC_GRID_S6), N_SEEDS))\n"
    "    t0 = time.time()\n"
    "    for ni, N in enumerate(tqdm(N_GRID_S6, desc='N')):\n"
    "        for ti, Tc in enumerate(TC_GRID_S6):\n"
    "            phi = greenwave_offsets(int(N), int(Tc))\n"
    "            for ki, seed in enumerate(SEEDS):\n"
    "                np.random.seed(seed)\n"
    "                sim = TwoLaneNaSchWithLights(\n"
    "                    L=L, n_lanes=2, v_max=V_MAX, p_rand=P_RAND,\n"
    "                    p_chg=PCHG_DEFAULT, boundary='open', p_in=P_IN,\n"
    "                    N=int(N), T_cycle=int(Tc), T_green=int(Tc) // 2,\n"
    "                    phase_offsets=phi,\n"
    "                )\n"
    "                flow[ni, ti, ki] = measure_J(sim, T_WARMUP, T_MEASURE)\n"
    "    return dict(flow=flow, N_grid=N_GRID_S6, T_cycle_grid=TC_GRID_S6,\n"
    "                seeds=SEEDS, T_warmup=T_WARMUP, T_measure=T_MEASURE,\n"
    "                wall_clock=time.time() - t0)\n"
    "\n"
    "S6 = load_or_compute(f'{DATA_DIR}/04_heatmap_N_Tcycle.npz', compute_heatmap)\n"
    "print(f'wall-clock: {float(S6[\"wall_clock\"]):.1f} s')\n"
))

cells.append(nbf.v4.new_code_cell(
    "mean_flow = S6['flow'].mean(axis=-1)\n"
    "fig, ax = plt.subplots(figsize=(7, 5))\n"
    "im = ax.imshow(mean_flow, aspect='auto', origin='lower',\n"
    "                cmap='viridis')\n"
    "ax.set_xticks(np.arange(len(S6['T_cycle_grid'])))\n"
    "ax.set_xticklabels(S6['T_cycle_grid'])\n"
    "ax.set_yticks(np.arange(len(S6['N_grid'])))\n"
    "ax.set_yticklabels(S6['N_grid'])\n"
    "ax.set_xlabel('T_cycle')\n"
    "ax.set_ylabel('N')\n"
    "ax.set_title('§6 — J heatmap (greenwave, green = T_cycle/2)')\n"
    "argmax = np.unravel_index(np.argmax(mean_flow), mean_flow.shape)\n"
    "ax.scatter([argmax[1]], [argmax[0]], marker='*', s=200,\n"
    "            color='red', edgecolor='white', linewidth=1.5,\n"
    "            label=f'opt N={int(S6[\"N_grid\"][argmax[0]])}, '\n"
    "                  f'Tc={int(S6[\"T_cycle_grid\"][argmax[1]])}')\n"
    "fig.colorbar(im, ax=ax, label='J')\n"
    "ax.legend(loc='upper right')\n"
    "fig.savefig(f'{FIGURES_DIR}/04_heatmap_N_Tcycle.png')\n"
    "plt.show()\n"
    "print(f'argmax: N={int(S6[\"N_grid\"][argmax[0]])}, '\n"
    "      f'T_cycle={int(S6[\"T_cycle_grid\"][argmax[1]])}, '\n"
    "      f'J={mean_flow[argmax]:.4f}')\n"
))

# ── Section 7 ──
cells.append(nbf.v4.new_markdown_cell(
    "## §7 — Sensitivity to p_rand\n\n"
    "Sweep `p_rand ∈ {0.0, 0.1, 0.2, 0.3, 0.5}`. Fixed `N=2, T_cycle=60, "
    "T_green=30`, greenwave offset (recomputed at each p_rand because "
    "v_free shifts with p_rand)."
))

cells.append(nbf.v4.new_code_cell(
    "PRAND_GRID = np.array([0.0, 0.1, 0.2, 0.3, 0.5])\n"
    "T_CYCLE_S7 = 60\n"
    "T_GREEN_S7 = 30\n"
    "N_S7 = 2\n"
    "\n"
    "def compute_J_vs_prand():\n"
    "    flow = np.zeros((len(PRAND_GRID), N_SEEDS))\n"
    "    t0 = time.time()\n"
    "    for pi, pr in enumerate(tqdm(PRAND_GRID, desc='p_rand')):\n"
    "        v_free_local = max(0.1, V_MAX - pr)\n"
    "        phi = np.array([int(round((i * (L / (N_S7 + 1)) / v_free_local) % T_CYCLE_S7))\n"
    "                        for i in range(N_S7)])\n"
    "        for ki, seed in enumerate(SEEDS):\n"
    "            np.random.seed(seed)\n"
    "            sim = TwoLaneNaSchWithLights(\n"
    "                L=L, n_lanes=2, v_max=V_MAX, p_rand=float(pr),\n"
    "                p_chg=PCHG_DEFAULT, boundary='open', p_in=P_IN,\n"
    "                N=N_S7, T_cycle=T_CYCLE_S7, T_green=T_GREEN_S7,\n"
    "                phase_offsets=phi,\n"
    "            )\n"
    "            flow[pi, ki] = measure_J(sim, T_WARMUP, T_MEASURE)\n"
    "    return dict(flow=flow, prand_grid=PRAND_GRID, seeds=SEEDS,\n"
    "                T_warmup=T_WARMUP, T_measure=T_MEASURE,\n"
    "                wall_clock=time.time() - t0)\n"
    "\n"
    "S7 = load_or_compute(f'{DATA_DIR}/04_J_vs_prand.npz', compute_J_vs_prand)\n"
    "print(f'wall-clock: {float(S7[\"wall_clock\"]):.1f} s')\n"
))

cells.append(nbf.v4.new_code_cell(
    "fig, ax = plt.subplots()\n"
    "m = S7['flow'].mean(axis=-1)\n"
    "s = S7['flow'].std(axis=-1)\n"
    "ax.plot(S7['prand_grid'], m, '-o')\n"
    "ax.fill_between(S7['prand_grid'], m - s, m + s, alpha=0.2)\n"
    "ax.set_xlabel('p_rand')\n"
    "ax.set_ylabel('Throughput J (cars / timestep)')\n"
    "ax.set_title('§7 — J vs p_rand')\n"
    "fig.savefig(f'{FIGURES_DIR}/04_J_vs_prand.png')\n"
    "plt.show()\n"
))

# ── Section 8 ──
cells.append(nbf.v4.new_markdown_cell(
    "## §8 — Sensitivity to p_chg\n\n"
    "Sweep `p_chg ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 1.0}`. Fixed `N=2, "
    "T_cycle=60, T_green=30`, greenwave offset. Reference horizontal at "
    "`p_chg = 0` shows the no-lane-change baseline."
))

cells.append(nbf.v4.new_code_cell(
    "PCHG_GRID = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 1.0])\n"
    "T_CYCLE_S8 = 60\n"
    "T_GREEN_S8 = 30\n"
    "N_S8 = 2\n"
    "\n"
    "def compute_J_vs_pchg():\n"
    "    flow = np.zeros((len(PCHG_GRID), N_SEEDS))\n"
    "    t0 = time.time()\n"
    "    phi = greenwave_offsets(N_S8, T_CYCLE_S8)\n"
    "    for ci, pc in enumerate(tqdm(PCHG_GRID, desc='p_chg')):\n"
    "        for ki, seed in enumerate(SEEDS):\n"
    "            np.random.seed(seed)\n"
    "            sim = TwoLaneNaSchWithLights(\n"
    "                L=L, n_lanes=2, v_max=V_MAX, p_rand=P_RAND,\n"
    "                p_chg=float(pc), boundary='open', p_in=P_IN,\n"
    "                N=N_S8, T_cycle=T_CYCLE_S8, T_green=T_GREEN_S8,\n"
    "                phase_offsets=phi,\n"
    "            )\n"
    "            flow[ci, ki] = measure_J(sim, T_WARMUP, T_MEASURE)\n"
    "    return dict(flow=flow, pchg_grid=PCHG_GRID, seeds=SEEDS,\n"
    "                T_warmup=T_WARMUP, T_measure=T_MEASURE,\n"
    "                wall_clock=time.time() - t0)\n"
    "\n"
    "S8 = load_or_compute(f'{DATA_DIR}/04_J_vs_pchg.npz', compute_J_vs_pchg)\n"
    "print(f'wall-clock: {float(S8[\"wall_clock\"]):.1f} s')\n"
))

cells.append(nbf.v4.new_code_cell(
    "fig, ax = plt.subplots()\n"
    "m = S8['flow'].mean(axis=-1)\n"
    "s = S8['flow'].std(axis=-1)\n"
    "ax.plot(S8['pchg_grid'], m, '-o')\n"
    "ax.fill_between(S8['pchg_grid'], m - s, m + s, alpha=0.2)\n"
    "ax.axhline(m[0], linestyle='--', color='k', alpha=0.5,\n"
    "           label=f'p_chg=0 baseline = {m[0]:.3f}')\n"
    "ax.set_xlabel('p_chg')\n"
    "ax.set_ylabel('Throughput J (cars / timestep)')\n"
    "ax.set_title('§8 — J vs p_chg')\n"
    "ax.legend()\n"
    "fig.savefig(f'{FIGURES_DIR}/04_J_vs_pchg.png')\n"
    "plt.show()\n"
))

# ── Section 9 ──
cells.append(nbf.v4.new_markdown_cell(
    "## §9 — Robustness of the optimum to p_chg\n\n"
    "Recompute the §6 grid at `p_chg ∈ {0, 0.3}` only. Report the "
    "difference heatmap; small magnitudes confirm the optimum is robust "
    "to lane-changing."
))

cells.append(nbf.v4.new_code_cell(
    "PCHG_PAIR = np.array([0.0, 0.3])\n"
    "\n"
    "def compute_robustness():\n"
    "    flow = np.zeros((len(N_GRID_S6), len(TC_GRID_S6), len(PCHG_PAIR), N_SEEDS))\n"
    "    t0 = time.time()\n"
    "    for ni, N in enumerate(tqdm(N_GRID_S6, desc='N')):\n"
    "        for ti, Tc in enumerate(TC_GRID_S6):\n"
    "            phi = greenwave_offsets(int(N), int(Tc))\n"
    "            for pi, pc in enumerate(PCHG_PAIR):\n"
    "                for ki, seed in enumerate(SEEDS):\n"
    "                    np.random.seed(seed)\n"
    "                    sim = TwoLaneNaSchWithLights(\n"
    "                        L=L, n_lanes=2, v_max=V_MAX, p_rand=P_RAND,\n"
    "                        p_chg=float(pc), boundary='open', p_in=P_IN,\n"
    "                        N=int(N), T_cycle=int(Tc), T_green=int(Tc) // 2,\n"
    "                        phase_offsets=phi,\n"
    "                    )\n"
    "                    flow[ni, ti, pi, ki] = measure_J(sim, T_WARMUP, T_MEASURE)\n"
    "    return dict(flow=flow, N_grid=N_GRID_S6, T_cycle_grid=TC_GRID_S6,\n"
    "                pchg_pair=PCHG_PAIR, seeds=SEEDS,\n"
    "                T_warmup=T_WARMUP, T_measure=T_MEASURE,\n"
    "                wall_clock=time.time() - t0)\n"
    "\n"
    "S9 = load_or_compute(f'{DATA_DIR}/04_robustness_pchg.npz', compute_robustness)\n"
    "print(f'wall-clock: {float(S9[\"wall_clock\"]):.1f} s')\n"
))

cells.append(nbf.v4.new_code_cell(
    "delta = S9['flow'][:, :, 1, :].mean(axis=-1) - S9['flow'][:, :, 0, :].mean(axis=-1)\n"
    "fig, ax = plt.subplots(figsize=(7, 5))\n"
    "vlim = float(np.max(np.abs(delta)))\n"
    "im = ax.imshow(delta, aspect='auto', origin='lower',\n"
    "                cmap='RdBu_r', vmin=-vlim, vmax=vlim)\n"
    "for i in range(len(S9['N_grid'])):\n"
    "    for j in range(len(S9['T_cycle_grid'])):\n"
    "        ax.text(j, i, f'{delta[i, j]:+.3f}', ha='center', va='center',\n"
    "                fontsize=9)\n"
    "ax.set_xticks(np.arange(len(S9['T_cycle_grid'])))\n"
    "ax.set_xticklabels(S9['T_cycle_grid'])\n"
    "ax.set_yticks(np.arange(len(S9['N_grid'])))\n"
    "ax.set_yticklabels(S9['N_grid'])\n"
    "ax.set_xlabel('T_cycle')\n"
    "ax.set_ylabel('N')\n"
    "ax.set_title('§9 — ΔJ heatmap: J(p_chg=0.3) − J(p_chg=0)')\n"
    "fig.colorbar(im, ax=ax, label='ΔJ')\n"
    "fig.savefig(f'{FIGURES_DIR}/04_robustness_pchg.png')\n"
    "plt.show()\n"
    "print(f'max |ΔJ| = {vlim:.4f}')\n"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Summary\n\n"
    "Notebook 04 is a homogeneous two-lane parity of `nagel_sensitivity.ipynb`. "
    "It sweeps the same axes (N, T_cycle, green fraction, v_max, p_rand) "
    "plus a two-lane-specific p_chg sweep, an N×T_cycle heatmap, and a "
    "robustness check. Cached arrays preserve per-seed shape so SD bands "
    "are recomputed at plot time."
))

nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

import sys
out = sys.argv[1] if len(sys.argv) > 1 else "04_two_lane_sensitivity.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"wrote {out} ({len(cells)} cells)")
