# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Traffic-flow cellular automaton simulations for a Mathematical Modelling II coursework. The repo is organised by team member: `Bakri/` holds Bakri's NaSch work (the current deliverable); `Ethan/` and `Aiden/` are placeholders for the other two members. Bakri's model code lives in `.py` modules inside `Bakri/notebooks/` (e.g. `Bakri/notebooks/nasch.py`, `Bakri/notebooks/nasch_two_lane.py`); Jupyter notebooks in the same directory import from those modules and run experiments. There is no `src/` package, no test suite, and no build step. The README.md is the authoritative reference for model definitions, parameter meanings, and expected results — consult it before modifying physics or interpretation.

## Running

```bash
pip install -r requirements.txt             # numpy, matplotlib, jupyter
jupyter notebook Bakri/notebooks/<name>.ipynb
```

Run notebooks top-to-bottom (**Kernel → Restart & Run All**). Individual sections are not independent — later sections depend on classes imported (or defined) in Section 1 and Section 2 of the same notebook. There is no "run a single test" workflow; the smallest reproducible unit is a whole notebook section.

Sweep-heavy notebooks are slow (5–15 min each). See the README "How to Run" table for per-notebook runtimes before kicking off a full re-run.

## Architecture

### Module-per-model convention

**Model classes live in `.py` modules inside `Bakri/notebooks/`; notebooks in the same directory import from them and run experiments.** The in-progress refactor covers the NaSch family first:

- `Bakri/notebooks/nasch.py` — `TrafficCA`, `TrafficCAWithLights`, `TrafficCA_Periodic`, `fundamental_diagram`.
- `Bakri/notebooks/nasch_two_lane.py` — `TwoLaneNaSch`, `TwoLaneNaSchWithLights`, `two_lane_fundamental_diagram` (imports `TrafficCAWithLights` from `nasch`).
- `Bakri/notebooks/nasch_intersection.py` — `IntersectionSystem` (perpendicular crossing of two single-lane roads) and the private `_RoadWithExternalLight` subclass of `TrafficCA` it composes.
- `Bakri/notebooks/nasch_two_lane_viz.py` — space–time recording and plotting helpers (`record_space_time`, `plot_space_time_velocity` / `_identity` / `_type`); used by `05_two_lane_mixed.ipynb`.

Notebooks that consume these modules must enable autoreload in their first code cell so edits to the `.py` files propagate without a kernel restart:

```python
%load_ext autoreload
%autoreload 2
```

**Exception — `nagel_schreckenberg.ipynb` remains self-contained.** This is the original coursework submission and keeps its own inline copy of the single-lane classes for submission integrity. Its logic must remain bit-for-bit identical to `nasch.py`; when you change one, update the other. Do not add imports from `nasch.py` to this notebook.

The BML-family notebooks (`Bakri/notebooks/alternative/biham_middleton_levine.ipynb`, `Bakri/notebooks/alternative/biham_sensitivity.ipynb`, `Bakri/notebooks/alternative/stochastic_bml.ipynb`, `Bakri/notebooks/alternative/two_intersection_bml.ipynb`, `Bakri/notebooks/alternative/grid_network_bml.ipynb`, and the `Bakri/notebooks/alternative/simulations/` animations) live under `Bakri/notebooks/alternative/` because they sit outside the CW2 brief; they still hold inline copies of their model classes. They will migrate to `.py` modules the same way; until they do, fixing a bug in a BML-family class requires `Grep`ing the class name (`class BML:`, `class StochasticBML:`, …) across `Bakri/notebooks/alternative/` and editing every copy.

### Class taxonomy

The notebooks follow a consistent inheritance pattern:

- **NaSch single-lane (1-D)**: `TrafficCA` (open boundary) → `TrafficCAWithLights`. A separate `TrafficCA_Periodic` exists for fundamental-diagram measurement.
- **NaSch two-lane (1-D × 2)**: `TwoLaneNaSch` (handles both periodic and open boundaries via `boundary=` flag) → `TwoLaneNaSchWithLights`. The lights subclass **delegates** to a private `TrafficCAWithLights` instance instead of reimplementing the schedule, so single-lane and two-lane runs with the same `(N, T_cycle, T_green, offset)` see an identical red/green sequence by construction.
- **NaSch perpendicular intersection**: `IntersectionSystem` composes two `_RoadWithExternalLight` instances (one per direction) crossing at cell `L // 2`. It does **not** subclass `TrafficCA` — it owns two of them. Lights are toggled externally each step from a single boolean (`phase < T_g1`), making mutual exclusion structural.
- **BML (2-D)**: `BML` → `BMLWithLights` (signalised intersection grid, global alternating rule).
- **Stochastic BML**: `BML` → `StochasticBML` (overrides movement methods to add dawdling); `StochasticBMLWithLights` inherits from `StochasticBML`.
- **Road-network BML**: `TwoIntersectionBML` and `GridNetworkBML` are standalone — they do **not** inherit from `BML` because their road topology (masked non-road cells, per-junction independent phase counters) is incompatible with BML's uniform-grid global alternating rule.

When subclassing, preserve the contract that `p_rand = 0` in `StochasticBML` reproduces `BML` exactly (there is a sanity-check cell asserting this).

### Measurement protocol

All flow measurements use the same two-phase protocol, with constants defined in Section 1 of each notebook:

1. **Warmup** (`T_WARMUP` steps): run the CA and discard; allows the system to reach statistical steady state from an initial random configuration.
2. **Measurement** (`T_MEASURE` steps): run the CA and average a flow observable.

Default values vary by model (see each notebook's Section 1). The warmup is especially important for BML, which can take hundreds of steps to settle into stripes or gridlock. Reducing `T_WARMUP` for speed will silently corrupt phase-transition plots.

### Traffic-light mechanism

Three distinct mechanisms coexist in the codebase — do not conflate them:

- **NaSch (single- and two-lane) / uniform-grid BML**: Lights are "phantom obstacles" placed at evenly-spaced cells. A red light behaves identically to a stationary car at that cell in the braking rule. One cycle controls all lights, with an optional phase offset per light for green-wave coordination. In two-lane NaSch, lights span both lanes simultaneously (per-lane signals are not supported).
- **Perpendicular intersection (`IntersectionSystem`)**: A single boolean per step decides which road is green, derived from `phase = time % T` and `T_g1 = round(eta · T)`. The two `_RoadWithExternalLight` instances each treat the crossing cell as a phantom obstacle iff their externally-set `is_red` flag is on. Mutual exclusion is **structural** — exactly one boolean is true.
- **Two-Intersection / Grid Network BML**: Each junction has a **fully independent** phase counter `phase_ij(t) = (t + delta_ij) mod T_cycle`. Half the cycle is horizontal-green/vertical-red, the other half is flipped. A car on one road is only affected by the light at the intersection it is approaching.

Grid Network BML supports three `offset_mode` strategies: `'sync'`, `'wave'` (diagonal green-wave), and `'random'`. The wave spacing formula `s = L/(N+1)` matches car travel time between intersections — changing it breaks the green-wave property.

### Numbered notebooks and the cache/figure pattern

The numbered notebooks `01_two_lane_baseline`, `02_two_lane_lights`, `03_intersection_baseline`, `04_two_lane_sensitivity`, `05_two_lane_mixed` each follow the same convention:

- **Caching.** Each defines an inline `load_or_compute(path, compute_fn, force=False)` helper. First run of a sweep cell calls `compute_fn()`, writes a `.npz` to `Bakri/data/<id>.npz`, and returns it; later runs load the cache. Pass `force=True` at the call site to recompute (e.g. after bumping `n_seeds`, `T_measure`, or changing physics).
- **Figures** are written to `Bakri/figures/<id>.png` from the same notebook. Convention: commit figures, do **not** commit data caches.
- **Half-size sweep parameters.** Section 1 ships with reduced `n_seeds` / `T_measure` for fast iteration; the publication-quality settings are in trailing `# full: …` comments on the same line. Restore those before producing report-quality output, and bump the cache `force=True` so the half-size `.npz` doesn't shadow the result.

A regression test exists at `Bakri/test_two_lane_regression.py`: it asserts that `TwoLaneNaSch(n_lanes=1, p_chg=0)` produces a bit-exact match to the archival single-lane fundamental diagram. It imports classes from `nagel_schreckenberg.ipynb` via the `Bakri/_nasch_nb.py` shim, so don't rename either side without updating the other. Run with `python Bakri/test_two_lane_regression.py`.

### Simulation/animation notebooks

The animation notebooks are split by model family:

- `Bakri/notebooks/simulations/nasch_animation.ipynb` — NaSch single-lane animation (alongside the main NaSch notebooks).
- `Bakri/notebooks/alternative/simulations/bml_animation.ipynb`, `stochastic_bml_animation.ipynb` — BML-family animations (alongside the alternative-track BML notebooks).

These are large (20–35 MB each) because they embed pre-rendered animation output. They duplicate the model class definitions from the main notebooks — keep them in sync when changing core model behavior.

## Conventions

- **Model classes live in `.py` modules inside `Bakri/notebooks/`; notebooks import from them.** Do not paste model classes back into notebooks. The one exception is `nagel_schreckenberg.ipynb` (see above). Keep the `.py` modules alongside the notebooks that use them so `from nasch import …` resolves without any `sys.path` manipulation — do not introduce a deeper package layer between member folder and notebooks.
- **Do not create new notebooks or `.md` files** unless the user explicitly asks. The notebook set is frozen by the coursework scope.
- **All physics parameters** (`L`, `v_max`, `p_rand`, `T_WARMUP`, `T_MEASURE`, `T_cycle`, `N`) are module-level constants at the top of each notebook's Section 1. Do not hardcode numeric values inside class methods or sweep loops — read them from the Section 1 constants so a single edit changes everything downstream.
- **Sweep cells are cached.** When you change physics, parameters, or `compute_fn` body in a numbered notebook, you must either pass `force=True` or delete the matching `Bakri/data/<id>.npz` — otherwise the next run silently returns the stale cache.
- **The LaTeX report lives at `Bakri/Report/main.tex`** (with sections under `Bakri/Report/main_sections/` and the `.bib` at `Bakri/Report/references.bib`). Figures referenced by the report come from `Bakri/figures/`; do not duplicate plotting logic into the report tree.
