# 25MAP211 Coursework 2 — Traffic flow with NaSch cellular automaton

Coursework submission for **25MAP211 Mathematical Modelling II** (Loughborough University). This is coursework, not a research project — scope is bounded by the brief and the marking rubric. The brief is **Modelling Group 2**: a single-lane one-way road with equidistant traffic lights, optimise through-flow as a function of the number of lights, the cycle length, and the green fraction. The model family is the **Nagel–Schreckenberg stochastic cellular automaton** (single-lane NaSch plus extensions). Beyond the baseline single-lane problem, two extension directions are explored: (i) a **two-lane NaSch** with CWS symmetric lane-changing and equidistant lights spanning both lanes, and (ii) a **perpendicular intersection** of two single-lane roads sharing one crossing cell with mutually-exclusive traffic lights.

## Repository structure

```
.
├── notebooks/
│   ├── nasch.py                         # TrafficCA, TrafficCAWithLights, TrafficCA_Periodic, fundamental_diagram()
│   ├── nasch_two_lane.py                # TwoLaneNaSch, TwoLaneNaSchWithLights, two_lane_fundamental_diagram()
│   ├── nasch_intersection.py            # IntersectionSystem (perpendicular crossing, mutually-exclusive lights)
│   │
│   ├── nagel_schreckenberg.ipynb        # Archival CW1 submission, self-contained; canonical NaSch + lights + space–time diagrams
│   ├── nagel_sensitivity.ipynb          # Single-lane parameter sweeps: N, T_cycle, green fraction, v_max
│   ├── nasch_two_lane.ipynb             # Working notebook; contains n_lanes=1 regression check
│   ├── 01_two_lane_baseline.ipynb       # Two-lane no-lights: periodic FD + open-boundary throughput
│   ├── 02_two_lane_lights.ipynb         # Two-lane with N=2 lights: (phase offset φ × lane-change p_chg) sweep
│   ├── 03_intersection_baseline.ipynb   # Perpendicular intersection: η sweep + asymmetric-demand heatmap
│   │
│   ├── biham_middleton_levine.ipynb     # Alternative 2-D models (see "Alternative models" section)
│   ├── biham_sensitivity.ipynb
│   ├── stochastic_bml.ipynb
│   ├── two_intersection_bml.ipynb
│   ├── grid_network_bml.ipynb
│   └── simulations/                     # Animation notebooks (20–35 MB each, embedded output)
│       ├── bml_animation.ipynb
│       ├── nasch_animation.ipynb
│       ├── stochastic_bml_animation.ipynb
│       └── two_intersection_bml.ipynb
│
├── data/                                # .npz caches from experiment cells (empty on fresh clone; not meant to be committed)
├── figures/                             # .png outputs from notebooks (empty until sweeps run; commit once generated)
├── requirements.txt
├── CLAUDE.md                            # Architecture / convention notes for the Claude Code assistant
└── README.md
```

`nagel_schreckenberg.ipynb` is the **archival CW1 submission**: it keeps its own inline copy of the single-lane classes for submission integrity and deliberately does not import from `nasch.py`. Its logic mirrors `nasch.py` bit-for-bit.

`data/` and `figures/` both exist but are empty on this machine — the sweep cells have not been run to completion yet. There is no `.gitignore` in the repo, so `data/*.npz` will be committed by default until one is added; the intended policy is **commit figures, do not commit data caches**.

## Running the code

- Python 3.10+ (numpy ≥ 2.0 requires this).
- Dependencies declared in `requirements.txt`: `numpy`, `matplotlib`, `jupyter`. **Additionally**: notebooks 01, 02, 03 import `tqdm` for progress bars; this is currently missing from `requirements.txt` and should be installed separately.

```bash
pip install -r requirements.txt
pip install tqdm
jupyter notebook notebooks/<name>.ipynb
```

Run each notebook top-to-bottom (Kernel → Restart & Run All). Later cells depend on constants and imports defined in Section 1, so individual sections are not independent.

Notebooks that import from the `.py` modules begin with

```python
%load_ext autoreload
%autoreload 2
```

so edits to `nasch.py`, `nasch_two_lane.py`, `nasch_intersection.py` propagate into the running kernel without a restart.

### Caching pattern

Each of the numbered notebooks (01, 02, 03) defines an inline helper

```python
def load_or_compute(path, compute_fn, force=False):
    if force or not os.path.exists(path):
        result = compute_fn()
        np.savez(path, **result)
        return result
    with np.load(path) as f:
        return {k: f[k] for k in f.files}
```

On the first run of an experiment cell the helper calls `compute_fn()`, writes the result to a `data/*.npz` cache, and returns it. On subsequent runs it loads the cache and returns immediately. Pass `force=True` at the call site to force recomputation (e.g. after bumping `n_seeds` or `T_measure`). Sweep-heavy notebooks ship with "half-size" parameters in Section 1 for rapid iteration; the publication-quality values appear in trailing `# full: …` comments.

## Model description

**Single-lane NaSch.** Cars live on a 1-D lattice of `L` cells; each car carries an integer velocity in `{0, …, v_max}`. One timestep applies the four-rule parallel update to every car (`nasch.py:123–133`): (1) accelerate: `v ← min(v+1, v_max)`; (2) slow for gap: `v ← min(v, gap)` where `gap` is the distance to the next obstacle (car or red light); (3) randomise: with probability `p_rand`, `v ← max(v−1, 0)`; (4) move: `x ← x + v`. Two boundary regimes are supported: **periodic** (`TrafficCA_Periodic`) for fundamental-diagram measurement at fixed car count, and **open** (`TrafficCA`) for through-flow with stochastic inflow at cell 0 (probability `p_in` per step) and outflow past cell `L−1`.

**Extensions.** (i) *Two-lane* (`TwoLaneNaSch` in `nasch_two_lane.py:21–323`): a `(2, L)` grid with a CWS symmetric lane-changing sub-step (`nasch_two_lane.py:140–173`) applied before the longitudinal update. A car switches with probability `p_chg` iff all four conditions hold — incentive `gap_own < v+1`, improvement `gap_other_ahead > gap_own`, adjacent cell empty, safe rear `gap_other_behind > v_max`. Per Rickert et al. (1996) and Chowdhury–Wolf–Schreckenberg (1997). (ii) *Equidistant lights* (`TrafficCAWithLights`, `TwoLaneNaSchWithLights`): `N` lights at cells `floor(L·(i+1)/(N+1))`; each light has phase `(time + phase_offsets[i]) mod T_cycle` and is red when `phase ≥ T_green`. A red light acts as a stationary obstacle at its cell for the braking rule. `offset=True` sets phase shifts `i · T_cycle / N` for a green-wave. `TwoLaneNaSchWithLights` delegates its red-light positions to a private `TrafficCAWithLights` instance (`nasch_two_lane.py:368–370, 385–387`), so the two-lane light schedule is single-lane by construction. (iii) *Perpendicular intersection* (`IntersectionSystem` in `nasch_intersection.py:51–122`): two independent `TrafficCA` roads share one crossing cell at `L//2`. A single boolean toggles which road is green, split by `eta = T_g1/T` (`T_g1 = round(eta·T)`, `T_g2 = T − T_g1`). Mutual exclusion is structural — both roads cannot be green simultaneously.

**Deliberately out of scope** (modelling choices, not oversights): no red-light violations; no multi-cell lane-change per timestep (the CWS rule is applied once per car per step); no vehicle heterogeneity (identical `v_max` and `p_rand` for every car); no learned or adaptive light controller.

## Validation

Checks that currently exist in the code or in a named notebook:

- **Canonical NaSch fundamental-diagram shape.** `fundamental_diagram()` (`nasch.py:293–308`) with `TrafficCA_Periodic` reproduces the published NaSch curve: linear free-flow branch of slope `v_max` at low density, universal jammed branch at high density. Run from `nagel_schreckenberg.ipynb` §4.
- **Two-lane → single-lane regression.** `two_lane_fundamental_diagram(n_lanes=1, p_chg=0)` (`nasch_two_lane.py:393–413`) should recover the single-lane curve; the constructor check at `nasch_two_lane.py:49–53` enforces `n_lanes ∈ {1, 2}`. Visualised overlay in `nasch_two_lane.ipynb` §5.
- **Light-oracle delegation.** `TwoLaneNaSchWithLights._red_light_positions()` calls through to a private `TrafficCAWithLights` instance each step, so red-light positions are identical between the single-lane and two-lane light models at every timestep — verified by construction rather than numerically.
- **Intersection symmetric demand.** With `p_in_1 = p_in_2`, `IntersectionSystem` should give `J_total(η)` symmetric about `η = 0.5`. `03_intersection_baseline.ipynb` runs a symmetric sweep and asserts `max|J_total(η) − J_total(1−η)|` lies within `3 × max seed std`, and that per-road throughput at `η = 0.5` is between 0.4 and 0.6 of the single-road baseline.

An analytical v_max=1 closed form (`J_max = (1 − √p)/2` at ρ=1/2) is **not** implemented yet and is not claimed as a passing validation here.

## Key figures

All figures below are produced by the numbered notebooks. `figures/` is empty on a fresh clone; the entries are annotated **(planned)** until the owning notebook has been run. Numbering matches what the report will use.

- Figure 1 — `figures/01_fd.png`, from `01_two_lane_baseline.ipynb` **(planned)**: two-lane periodic fundamental diagram for `p_chg ∈ {0, 0.25, 0.5, 1.0}`, with the single-lane curve overlaid as a dashed reference.
- Figure 2 — `figures/01_open.png`, from `01_two_lane_baseline.ipynb` **(planned)**: open-boundary throughput vs inflow `p_in` for `v_max ∈ {1, 2, 3, 5}` × `p_chg ∈ {0, 1}`, with the `y = p_in` line shown to quantify boundary-insertion failure.
- Figure 3 — `figures/02_phi_sweep.png`, from `02_two_lane_lights.ipynb` **(planned)**: throughput vs phase offset φ, one curve per `p_chg`, with in-phase, anti-phase, and empirical green-wave offsets marked.
- Figure 4 — `figures/02_heatmap.png`, from `02_two_lane_lights.ipynb` **(planned)**: 2-D heatmap of throughput over (φ, `p_chg`) at fixed `p_in = 0.4`, `N = 2`, `T_cycle = 60`.
- Figure 5 — `figures/03_symmetric.png`, from `03_intersection_baseline.ipynb` **(planned)**: per-road and total throughput (J1, J2, J_total) vs green-time split η, with the symmetry check about η = 0.5.
- Figure 6 — `figures/03_asym_heatmap.png`, from `03_intersection_baseline.ipynb` **(planned)**: J_total heatmap over asymmetric demand `(p_in_1, p_in_2)` at η = 0.5, with contour overlays.

## Parameter conventions

| Symbol | Code name | Meaning | Typical value |
|---|---|---|---|
| L | `L` | Road length (cells per lane) | 500 (1000 for `02_two_lane_lights`) |
| v_max | `v_max` | Velocity cap | 5 |
| p | `p_rand` | Dawdling (randomisation) probability | 0.3 |
| p_in | `p_in`, `p_in_1`, `p_in_2` | Inflow probability per lane per step | 0.3–0.5 |
| p_chg | `p_chg` | Lane-change probability (two-lane only) | 0 – 1 |
| N | `N` | Number of equidistant lights | 2 |
| T_cycle | `T_cycle` | Light cycle length (steps) | 30–80 |
| T_green | `T_green` | Green phase duration (steps) | `T_cycle // 2` |
| φ | `phase_offsets[i]` | Per-light phase offset (NaSch lights) | 0 or `T_cycle / N` |
| T | `T` | Intersection cycle length (steps) | 80 |
| η | `eta` | Green-time split for road 1: `T_g1 / T` | 0.1 – 0.9 |
| ρ | derived | Density = (total cars) / (`n_lanes · L`) | 0 – 1 |
| J | derived | Flow = `⟨v⟩ · ρ` (periodic) or `outflow / T` (open) | 0 – v_max/2 |
| T_WARMUP | `T_WARMUP` | Transient steps discarded before measurement | 500 (module default); 1000–5000 in 01/02/03 |
| T_MEASURE | `T_MEASURE` | Steps averaged after warmup | 1000 (module default); 2000–10000 in 01/02/03 |

Values are sourced from module defaults (`nasch.py:24–25, 40–49`; `nasch_two_lane.py:38–47`; `nasch_intersection.py:75–92`) and the Section 1 constants of each notebook.

## Alternative models (briefly)

The repository also contains exploratory work on 2-D cellular automata that are **outside the CW2 brief** and retained for comparative context only: the deterministic Biham–Middleton–Levine model (`biham_middleton_levine.ipynb`, `biham_sensitivity.ipynb`), a stochastic variant that reintroduces NaSch-style dawdling on a 2-D grid (`stochastic_bml.ipynb`), and BML on explicit road networks with per-junction lights (`two_intersection_bml.ipynb`, `grid_network_bml.ipynb`). The notebooks in `notebooks/simulations/` are animation-only companions (20–35 MB each, with embedded output). The CW2 analysis and the figures referenced above are confined to the NaSch-family notebooks.

## Known limitations

- **Performance.** The periodic and open longitudinal updates in `nasch_two_lane.py` use per-car Python loops for gap computation. Full-resolution sweeps (`n_seeds=5`, `T_measure=10000`) take several minutes per notebook. Notebooks 01 and 02 ship with half-size parameters (`n_seeds=3`, smaller `T_measure`) and `# full: …` comments indicating the publication-quality settings.
- **Lane count.** `TwoLaneNaSch` hard-codes `n_lanes ∈ {1, 2}` at `nasch_two_lane.py:49–53`; supporting three or more lanes would require reworking the lane-change indexing.
- **Shared light across lanes.** Lights span both lanes simultaneously (suitable for a shared-phase junction; per-lane signals are not supported).
- **Single crossing in `IntersectionSystem`.** One perpendicular crossing only; there is no N × M grid generalisation inside the NaSch-family modules (the BML-family `grid_network_bml.ipynb` covers that separately, but with a different physical model).
- **Sensitivity to `p_rand`** has not been systematically mapped for the two-lane-with-lights configuration; a notebook 04 is planned but not present.
- **Repo hygiene.** `requirements.txt` omits `tqdm` despite the new notebooks importing it; no `.gitignore` file exists, so `data/*.npz` caches will be committed by default until one is added.

## References

- Nagel, K. & Schreckenberg, M. (1992). A cellular automaton model for freeway traffic. *Journal de Physique I*, 2, 2221–2229.
- Rickert, M., Nagel, K., Schreckenberg, M. & Latour, A. (1996). Two lane traffic simulations using cellular automata. *Physica A*, 231, 534–550.
- Chowdhury, D., Wolf, D. E. & Schreckenberg, M. (1997). Particle hopping models for two-lane traffic with two kinds of vehicles: effects of lane-changing rules. *Physica A*, 235, 417–439.

## Author

Bakri Othman — 25MAP211 Mathematical Modelling II, Loughborough University, 2025/26.
