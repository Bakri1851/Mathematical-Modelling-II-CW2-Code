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

Notebooks that consume these modules must enable autoreload in their first code cell so edits to the `.py` files propagate without a kernel restart:

```python
%load_ext autoreload
%autoreload 2
```

**Exception — `nagel_schreckenberg.ipynb` remains self-contained.** This is the original coursework submission and keeps its own inline copy of the single-lane classes for submission integrity. Its logic must remain bit-for-bit identical to `nasch.py`; when you change one, update the other. Do not add imports from `nasch.py` to this notebook.

The BML-family notebooks (`Bakri/notebooks/alternative/biham_middleton_levine.ipynb`, `Bakri/notebooks/alternative/biham_sensitivity.ipynb`, `Bakri/notebooks/alternative/stochastic_bml.ipynb`, `Bakri/notebooks/alternative/two_intersection_bml.ipynb`, `Bakri/notebooks/alternative/grid_network_bml.ipynb`, and the `Bakri/notebooks/alternative/simulations/` animations) live under `Bakri/notebooks/alternative/` because they sit outside the CW2 brief; they still hold inline copies of their model classes. They will migrate to `.py` modules the same way; until they do, fixing a bug in a BML-family class requires `Grep`ing the class name (`class BML:`, `class StochasticBML:`, …) across `Bakri/notebooks/alternative/` and editing every copy.

### Class taxonomy

The notebooks follow a consistent inheritance pattern:

- **NaSch (1-D)**: `TrafficCA` (open boundary) → `TrafficCAWithLights`. A separate `TrafficCA_Periodic` exists for fundamental-diagram measurement.
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

Two distinct mechanisms coexist in the codebase — do not conflate them:

- **NaSch / uniform-grid BML**: Lights are "phantom obstacles" placed at evenly-spaced cells. A red light behaves identically to a stationary car at that cell in the braking rule. One cycle controls all lights, with an optional phase offset per light for green-wave coordination.
- **Two-Intersection / Grid Network BML**: Each junction has a **fully independent** phase counter `phase_ij(t) = (t + delta_ij) mod T_cycle`. Half the cycle is horizontal-green/vertical-red, the other half is flipped. A car on one road is only affected by the light at the intersection it is approaching.

Grid Network BML supports three `offset_mode` strategies: `'sync'`, `'wave'` (diagonal green-wave), and `'random'`. The wave spacing formula `s = L/(N+1)` matches car travel time between intersections — changing it breaks the green-wave property.

### Simulation/animation notebooks

The animation notebooks are split by model family:

- `Bakri/notebooks/simulations/nasch_animation.ipynb` — NaSch single-lane animation (alongside the main NaSch notebooks).
- `Bakri/notebooks/alternative/simulations/bml_animation.ipynb`, `stochastic_bml_animation.ipynb` — BML-family animations (alongside the alternative-track BML notebooks).

These are large (20–35 MB each) because they embed pre-rendered animation output. They duplicate the model class definitions from the main notebooks — keep them in sync when changing core model behavior.

## Conventions

- **Model classes live in `.py` modules inside `Bakri/notebooks/`; notebooks import from them.** Do not paste model classes back into notebooks. The one exception is `nagel_schreckenberg.ipynb` (see above). Keep the `.py` modules alongside the notebooks that use them so `from nasch import …` resolves without any `sys.path` manipulation — do not introduce a deeper package layer between member folder and notebooks.
- **Do not create new notebooks or `.md` files** unless the user explicitly asks. The notebook set is frozen by the coursework scope.
- **All physics parameters** (`L`, `v_max`, `p_rand`, `T_WARMUP`, `T_MEASURE`, `T_cycle`, `N`) are module-level constants at the top of each notebook's Section 1. Do not hardcode numeric values inside class methods or sweep loops — read them from the Section 1 constants so a single edit changes everything downstream.
