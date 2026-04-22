# 25MAP211 Coursework 2 — Traffic flow with NaSch cellular automaton

Coursework submission for **25MAP211 Mathematical Modelling II** (Loughborough University). This is coursework, not a research project — scope is bounded by the brief and the marking rubric. The brief is **Modelling Group 2**: a single-lane one-way road with equidistant traffic lights, optimise through-flow as a function of the number of lights, the cycle length, and the green fraction. The model family is the **Nagel–Schreckenberg stochastic cellular automaton** (single-lane NaSch plus extensions). Beyond the baseline single-lane problem, two extension directions are explored: (i) a **two-lane NaSch** with CWS symmetric lane-changing and equidistant lights spanning both lanes, and (ii) a **perpendicular intersection** of two single-lane roads sharing one crossing cell with mutually-exclusive traffic lights

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

### Single-lane NaSch

The road is a 1-D lattice of `L` cells. Each cell is either empty or occupied by exactly one car, and every car carries an integer velocity `v ∈ {0, 1, …, v_max}` — so cells per timestep is the natural unit of speed. The state of the simulation is `self.road`: an integer array of length `L` where `-1` encodes "empty" and any non-negative value encodes a car's current velocity (`nasch.py:40–49`).

One timestep is a **parallel update** of every car using the four NaSch rules (`nasch.py:123–133`), applied in this order:

1. **Accelerate** — `v ← min(v + 1, v_max)`. Drivers always want to go faster, capped at the speed limit.
2. **Slow for the gap** — `v ← min(v, gap)`, where `gap` is the number of empty cells between this car and the next obstacle ahead (the next car or a red light). This guarantees no collisions.
3. **Randomise (dawdling)** — with probability `p_rand`, `v ← max(v − 1, 0)`. This single stochastic kick is what generates the spontaneous stop-and-go waves that make NaSch more than a deterministic model.
4. **Move** — `x ← x + v`.

All four rules are evaluated from the *same* pre-step state and applied simultaneously, not sequentially per car. Obstacle positions are assembled once per step as the sorted union of car positions and red-light positions (`nasch.py:117–119`); gaps are computed with `np.searchsorted` against this list (`nasch_two_lane.py:246–249` uses the same idea for open boundaries).

Two boundary regimes coexist:

- **Open** (`TrafficCA`, `nasch.py:28–184`) — used whenever through-flow is the observable. At each step, a new car is injected into cell 0 with velocity 0 and probability `p_in`, provided cell 0 is empty (`nasch.py:155–160`). Any car whose new position `x + v ≥ L` exits the road and increments `outflow_count`. Steady-state flow is measured as `outflow_count / T_measure` (`nasch.py:170–176`).
- **Periodic** (`TrafficCA_Periodic`, `nasch.py:237–290`) — used for fundamental-diagram measurement at fixed car count. Positions wrap modulo `L`, no inflow/outflow, and flow is computed as `J = ⟨v⟩ · ρ`, averaged over `T_measure` steps.

**Traffic lights** (`TrafficCAWithLights`, `nasch.py:186–234`) extend the open-boundary model with `N` equidistant lights at cells `floor(L·(i+1)/(N+1))`. Each light `i` has its own phase counter `(time + phase_offsets[i]) mod T_cycle`; the light is green while that phase is below `T_green` and red otherwise. A red light appears in the obstacle list for that step, so Rule 2 treats it exactly like a stationary car at its cell — no new physics, just an extra obstacle. Setting `offset=True` gives consecutive lights a phase shift of `T_cycle / N`, producing a green-wave. All of this is inherited by the two-lane model via delegation (see below), so a single-lane and two-lane simulation using the same `N`, `T_cycle`, `T_green`, `offset` see an identical red/green schedule.

### Two-lane NaSch

State is now a `(n_lanes, L)` grid — effectively two single-lane roads stacked, with `n_lanes` restricted to `{1, 2}` by the constructor (`nasch_two_lane.py:49–53`). One timestep (`nasch_two_lane.py:280–294`) is three sub-steps executed in order:

1. **Lane-change sub-step** (`_lane_change_substep`, `nasch_two_lane.py:140–173`) — skipped entirely when `n_lanes = 1` or `p_chg = 0`. For each lane, every car computes four gaps: the gap ahead in its own lane (`gap_own`), the gap ahead in the other lane (`gap_other_ahead`), the gap to the nearest car *behind* in the other lane (`gap_other_behind`), and whether the adjacent cell in the other lane is currently empty. Following the CWS symmetric rule, the car switches lanes iff **all four** conditions hold:
   - *Incentive:* its own lane is too tight — `gap_own < v + 1`.
   - *Improvement:* the other lane is actually better — `gap_other_ahead > gap_own`.
   - *Adjacent cell empty:* the target cell is free right now.
   - *Safe rear:* the car behind in the target lane is at least `v_max` cells away, so it cannot rear-end the mover on the next longitudinal step.
   
   If all four hold, the car switches with probability `p_chg`. The rule is applied to both lanes from the *same* pre-step snapshot (a copy of `self.road` is taken at the top of the method, so swaps in lane 0 cannot influence swaps in lane 1), preserving the parallel-update invariant.

2. **Longitudinal sub-step** (per lane, inside the step loop at `nasch_two_lane.py:285–289`) — each lane runs the same four-rule NaSch update from §Single-lane NaSch. The obstacle list for a lane is the union of cars **in that lane** and **red-light cells**; lights span both lanes simultaneously (`nasch_two_lane.py:192–198` for periodic, `237–240` for open). Periodic and open boundaries use their own specialised methods (`_step_lane_periodic` and `_step_lane_open`).

3. **Inflow** (`_inflow`, `nasch_two_lane.py:271–276`, open boundary only) — one candidate car per lane at cell 0, each injected independently with probability `p_in`.

Two-lane lights are implemented as **delegation, not duplication**. `TwoLaneNaSchWithLights.__init__` constructs a private `TrafficCAWithLights` instance (`nasch_two_lane.py:368–370`), and `_red_light_positions()` simply syncs the oracle's clock to its own and returns the oracle's answer (`nasch_two_lane.py:385–387`). This makes it impossible for the two-lane light schedule to drift from the single-lane one — they are literally the same object.

The lane-changing rule follows Rickert, Nagel, Schreckenberg & Latour (1996) for two-lane CA traffic and the symmetric CWS variant described in Chowdhury, Wolf & Schreckenberg (1997).

### Perpendicular intersection

`IntersectionSystem` (`nasch_intersection.py:51–122`) models two single-lane open-boundary NaSch roads that cross at a single cell `L // 2`. Road 1 runs one direction, road 2 runs perpendicular; they never share cells except at the crossing. Internally each road is a `_RoadWithExternalLight` — a thin `TrafficCA` subclass whose `_red_light_positions()` returns `[crossing_cell]` iff its externally-toggled `is_red` flag is set (`nasch_intersection.py:31–48`).

One timestep (`nasch_intersection.py:101–106`) is:

1. **Set the light state.** Compute `phase = time % T`. Road 1 is green while `phase < T_g1`, road 2 is green otherwise, where `T_g1 = round(eta · T)` and `T_g2 = T − T_g1`. Set `road1.is_red` and `road2.is_red` from this single boolean (`nasch_intersection.py:95–103`). **Mutual exclusion is structural** — exactly one boolean is true, so both lights can never be green simultaneously.
2. **Advance each road independently.** Each road runs a full open-boundary NaSch step, with the crossing cell appearing in its obstacle list whenever its `is_red` flag is on. A car approaching a red crossing therefore brakes for an invisible stopped car sitting on `L // 2`, just like any other red light in this code base.
3. **Advance the clock** (`self.time += 1`).

The key knob is `eta`, the fraction of cycle time given to road 1. Throughput is measured independently for each road (`outflow_count / T` for both) and returned together with their sum (`nasch_intersection.py:114–122`):

```python
J1, J2, J_total = system.run(T_measure)
```

Under symmetric demand (`p_in_1 = p_in_2`), `J_total(η)` should be symmetric about `η = 0.5`, which is precisely the validation check run in `03_intersection_baseline.ipynb`.

### Deliberately out of scope

These are modelling choices bounded by the CW2 brief, not oversights:

- No red-light violations — the braking rule treats a red cell as impassable.
- No multi-cell lane-change per timestep — the CWS rule is applied once per car per step, so a car cannot traverse more than one lane in one step (two-lane only ever has one alternative anyway).
- No vehicle heterogeneity — every car has the same `v_max` and `p_rand`.
- No learned or adaptive light controller — all light schedules are fixed at construction time.

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
