# Mathematical Modelling 2 — CW2: Traffic Flow

Two cellular automaton models for traffic flow, compared side-by-side:

| Model | File | Type | Key phenomenon |
|---|---|---|---|
| **Nagel-Schreckenberg (NaSch)** | `notebooks/nagel_schreckenberg.ipynb` | 1-D stochastic | Stop-and-go waves; smooth fundamental diagram |
| **Biham-Middleton-Levine (BML)** | `notebooks/biham_middleton_levine.ipynb` | 2-D deterministic | Sharp phase transition; gridlock |
| **Stochastic BML** | `notebooks/stochastic_bml.ipynb` | 2-D stochastic | Softened transition; NaSch dawdling in 2-D |

---

## Problem Statement

A single-lane one-way street has several equidistant traffic lights. The goal is to optimise **through-flow** (cars passing per timestep) by tuning:
- Overall velocity limitation $v_\text{max}$
- Number of traffic lights $N$
- Switching cycle length $T_\text{cycle}$ and green fraction

The primary tool is a **discrete stochastic model** — a cellular automaton.

---

## Methodology

### Nagel-Schreckenberg (NaSch) Model

The road is represented as a 1D array of $L$ cells. Each cell is either empty or occupied by a car with an integer velocity $v \in \{0, 1, \ldots, v_\text{max}\}$.

At each timestep, **four rules are applied simultaneously** to every car:

| Step | Rule | Update |
|------|------|--------|
| 1 | **Acceleration** | $v_i \leftarrow \min(v_i + 1,\; v_\text{max})$ |
| 2 | **Braking** | $v_i \leftarrow \min(v_i,\; d_i - 1)$ where $d_i$ = gap to next obstacle |
| 3 | **Randomisation** | With probability $p$: $v_i \leftarrow \max(v_i - 1,\; 0)$ |
| 4 | **Movement** | $x_i \leftarrow x_i + v_i$ |

The randomisation step (dawdling) introduces stochasticity and is responsible for the spontaneous formation of **stop-and-go waves** — backward-propagating congestion bands observed in real traffic. With $p_\text{rand} = 0.3$, each car has a 30% chance of decelerating by 1 cell/step on any given timestep, even when the road ahead is clear. This mimics driver inattentiveness and is sufficient to nucleate phantom traffic jams at higher densities.

**Boundary conditions (open):** Cars enter cell 0 with probability $p_\text{in}$ and exit at cell $L$. Through-flow is measured as the count of cars leaving per timestep.

### Traffic Lights

$N$ lights are placed at evenly-spaced positions $\left\lfloor \frac{L (i+1)}{N+1} \right\rfloor$ for $i = 0, \ldots, N-1$.

Each light cycles between:
- **Green** phase: lasts $T_\text{green}$ timesteps (cars pass freely)
- **Red** phase: lasts $T_\text{red} = T_\text{cycle} - T_\text{green}$ timesteps

When a light is red, its cell acts as a **phantom obstacle** — the braking rule treats it identically to a stationary car ahead.

Two coordination strategies are compared:
- **Synchronised**: all lights in the same phase
- **Offset (green wave)**: light $i$ has phase shift $i \cdot T_\text{cycle}/N$, creating a coordinated wave that allows cars to pass through consecutive lights without stopping

---

## Notebook Structure

### `notebooks/nagel_schreckenberg.ipynb`

| Section | Description |
|---------|-------------|
| **1. Imports & Parameters** | Global constants (`L`, `v_max`, `p_rand`, warmup/measurement lengths) |
| **2. Core NaSch CA** | `TrafficCA` class — open boundary, vectorised update, inflow/outflow |
| **3. Traffic Lights** | `TrafficCAWithLights` — inherits from `TrafficCA`, adds $N$ lights with cycle and offset |
| **4. Fundamental Diagram** | Periodic boundary: flow $J$ vs density $\rho$ for multiple $v_\text{max}$ values |
| **5. Flow vs Inflow** | Open boundary: through-flow vs $p_\text{in}$ for multiple $v_\text{max}$ values |
| **6. Space-Time Diagrams** | Occupancy visualisations: free flow, congestion, and light effects |
| **7. Traffic Light Study** | Sweeps of $N$, $T_\text{cycle}$, green fraction, and synchronisation mode |
| **8. Optimisation Heatmap** | 2D sweep of $(N, T_\text{cycle})$ → optimal configuration identified |
| **9. Summary** | Comparative plots and key findings |

### `notebooks/nagel_sensitivity.ipynb`

Systematic parameter sensitivity study for the NaSch model:

| Section | Description |
|---------|-------------|
| **2. Number of lights N** | Monotonic flow decrease as N increases — each light is a bottleneck |
| **3. Cycle length T_cycle** | Flow peaks at an intermediate T_cycle; too short or too long both hurt |
| **4. Green fraction** | Optimal duty cycle near 50%; asymmetric splits reduce flow |
| **5. Maximum velocity** | Higher v_max raises capacity but lights impose a hard ceiling |
| **6. 2D heatmap** | Joint sweep of (N, T_cycle) for both sync and offset modes |
| **7. Summary comparison** | Flow vs p_in for no-lights, optimal sync, and optimal offset configurations |

### `notebooks/biham_middleton_levine.ipynb`

| Section | Description |
|---------|-------------|
| **1. Imports & Parameters** | Global constants (`L`, warmup/measurement lengths) |
| **2. BML CA** | `BML` class — 2-D periodic grid, vectorised simultaneous update |
| **3. Fundamental Diagram** | Flow vs density: reveals sharp phase transition at $\rho_c$ |
| **4. Phase Transition Analysis** | Order parameter vs $\rho$ for multiple $L$; susceptibility peak locates $\rho_c$ |
| **5. Grid Snapshots** | Colour-coded grid at free-flow, critical, and gridlock densities |
| **6. Space-Time Diagrams** | Row occupancy over time: free flow, intermittent jams, frozen gridlock |
| **7. Traffic Lights** | `BMLWithLights` — signalised intersection grid; FD comparison, cycle sweep, snapshots |
| **8. Summary** | Fundamental diagram + bar chart of key regimes |

### `notebooks/stochastic_bml.ipynb`

Bridges the deterministic BML and stochastic NaSch models by injecting a **NaSch-style
dawdling step** into the 2-D BML grid. The `StochasticBML` class inherits from `BML` and
overrides both movement methods: after checking that the target cell is empty, each car is
independently frozen with probability $p_{\text{rand}}$ before the move is applied.
Setting $p_{\text{rand}} = 0$ recovers the deterministic BML exactly — no code paths differ.
`StochasticBMLWithLights` further extends this with intersection-based signal control,
combining light blocking and stochastic dawdling in a single model.

| Section | Description |
|---------|-------------|
| **1. Imports & Parameters** | Global constants: `L=100`, `T_WARMUP=500`, `T_MEASURE=200`, `P_RAND=0.3` |
| **2. Model** | `BML` base class (self-contained copy) + `StochasticBML` subclass with dawdling rule; sanity check confirms `p_rand=0` is bit-for-bit identical to BML |
| **3. Fundamental Diagram** | Overlaid flow-vs-density curves for $p_{\text{rand}} \in \{0, 0.1, 0.3, 0.5\}$; shows how stochasticity smooths and shifts the sharp BML drop |
| **4. Phase Transition Analysis** | Order parameter and susceptibility $-\mathrm{d}\phi/\mathrm{d}\rho$ for each $p_{\text{rand}}$ at $L=128$; peak location gives $\rho_c(p)$; peak broadening quantifies softening |
| **5. Grid Snapshots** | 2×3 panel: deterministic ($p=0$) vs stochastic ($p=0.3$) rows, three density columns (free-flow / critical / gridlock); self-organised stripes visible in det., smeared in stochastic |
| **6. Space-Time Diagrams** | 2×3 row-occupancy panels; key observation: deterministic near-gridlock freezes permanently, stochastic shows intermittent gap openings — gridlock becomes probabilistic |
| **7. Effect of $p_{\text{rand}}$** | Flow vs $p_{\text{rand}}$ at three fixed densities; free-flow cost is monotone; near-gridlock density may show non-monotone benefit as stochastic unfreezing escapes absorbing states |
| **8. Traffic Lights** | `BMLWithLights` (verbatim copy) + `StochasticBMLWithLights`; four-scenario fundamental diagram isolates contributions of stochasticity and lights independently; grid snapshots with yellow intersection overlay |
| **9. Summary** | Overlaid fundamental diagrams with $\rho_c$ verticals + dual-axis bar chart of peak flow and $\rho_c$ vs $p_{\text{rand}}$ |

### `notebooks/biham_sensitivity.ipynb`

| Section | Description |
|---------|-------------|
| **1. Car Type Ratio** | Effect of horizontal/vertical imbalance on flow near $\rho_c$ |
| **2. System Size Effects** | Fundamental diagram for $L \in \{32, 64, 128, 256\}$ — finite-size scaling |
| **3. Summary** | Side-by-side summary plots |

### `notebooks/simulations/` — Watchable Animations

| Notebook | Animations |
|---------|------------|
| **`bml_animation.ipynb`** | Three density regimes side by side · Density sweep (watch the transition freeze) · Space–time diagram building up row by row · **Traffic lights live grid** (queue-and-release at intersections) |
| **`nasch_animation.ipynb`** | Free flow vs congestion · Stop-and-go waves on a periodic road · Traffic lights with synchronised vs offset (green wave) coordination |
| **`stochastic_bml_animation.ipynb`** | Three density regimes with $p_{\text{rand}}=0.3$ · $p_{\text{rand}}$ sweep at critical density (watch stripes dissolve) · Deterministic vs stochastic space-time comparison · **Traffic lights + dawdling** (combined effect) |

---

## Biham–Middleton–Levine (BML) Model

The BML model places two types of cars on a periodic $L \times L$ grid:
- **Horizontal cars** move one cell right each even sub-step.
- **Vertical cars** move one cell up each odd sub-step.

All moves in a sub-step are evaluated simultaneously on the current state and applied at once.
There is no stochastic element.

### Key parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `L` | 100 | Grid size ($L \times L$ cells) |
| `r_horiz` | 0.5 | Fraction of cars that are horizontal |
| `T_WARMUP` | 500 | Transient steps discarded |
| `T_MEASURE` | 200 | Steps averaged for flow |

### Phase transition

Unlike NaSch, BML exhibits a **sharp (discontinuous) phase transition** at a critical density
$\rho_c \approx 0.32$ (for $L = 100$):

- **Below $\rho_c$**: free flow — nearly all cars can move every step.
- **Near $\rho_c$**: self-organised diagonal stripe patterns; partial flow.
- **Above $\rho_c$**: permanent **gridlock** — an absorbing state that cannot be escaped.

The transition sharpens with $L$ (finite-size effect) and converges toward a step function as
$L \to \infty$.

### Contrast with NaSch

| Property | NaSch | BML |
|---|---|---|
| Dimension | 1-D | 2-D |
| Stochastic? | Yes ($p_\text{rand} = 0.3$) | No |
| Fundamental diagram | Smooth hump | Sharp drop |
| Congestion type | Backward-propagating stop-and-go waves | Irreversible total gridlock |
| Key parameter | $p_\text{rand}$, $v_\text{max}$ | $\rho$, grid size $L$ |

---

## Stochastic BML Model

The Stochastic BML model extends the deterministic BML with a **NaSch-style dawdling rule**
applied independently to each car during each movement phase. A car that would otherwise move
(target cell empty) is instead frozen with probability $p_{\text{rand}}$, mimicking driver
hesitation or reaction-time delay.

### Modified movement condition

Each car in each sub-step obeys:

| Condition | Outcome |
|-----------|---------|
| Target cell occupied | Car stays (collision avoidance — same as deterministic BML) |
| Target empty, $U_i > p_{\text{rand}}$ | Car moves (normal flow) |
| Target empty, $U_i \leq p_{\text{rand}}$ | Car stays (stochastic dawdling) |

where $U_i \sim \mathrm{Uniform}(0,1)$ is drawn independently for every car at every
sub-step. Both horizontal and vertical phases draw fresh samples. Setting $p_{\text{rand}} = 0$
makes the dawdling event impossible, recovering the deterministic BML exactly.

### Key parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `L` | 100 | Grid size ($L \times L$ cells) |
| `r_horiz` | 0.5 | Fraction of cars that are horizontal |
| `P_RAND` | 0.3 | Dawdling probability (matches NaSch default) |
| `T_WARMUP` | 500 | Transient steps discarded before measurement |
| `T_MEASURE` | 200 | Steps averaged for flow |
| `n_lights` | 4 | Number of lights per row/column in the intersection grid |
| `T_cycle` | 20 | Light cycle length (steps) |

### Effect on the phase transition

Adding $p_{\text{rand}} > 0$ changes the BML phase transition qualitatively:

- **Transition broadens**: the sharp discontinuous drop at $\rho_c$ becomes a gradual decline,
  resembling a smooth crossover rather than a first-order step.
- **$\rho_c$ shifts downward**: random hesitation reduces effective throughput even at low
  density, so the system reaches its capacity ceiling at a lower density.
- **Gridlock becomes probabilistic**: above the deterministic $\rho_c$, configurations that
  are permanent absorbing states in the deterministic model can now be partially escaped — a
  dawdling car occasionally creates the gap another car needs to unblock a cluster.
- **Free-flow cost**: at $\rho < \rho_c$, stochasticity always reduces flow because some cars
  skip moves they could have made freely.

### Three-way contrast

| Property | NaSch | BML | Stochastic BML |
|---|---|---|---|
| Dimension | 1-D | 2-D | 2-D |
| Stochastic? | Yes ($p_{\text{rand}} = 0.3$) | No | Yes ($p_{\text{rand}} > 0$) |
| Fundamental diagram | Smooth hump | Sharp discontinuous drop | Softened drop; shifts left with $p$ |
| Congestion type | Backward-propagating stop-and-go waves | Irreversible total gridlock | Probabilistic partial gridlock; rare unjamming events |
| Key parameter | $p_{\text{rand}}$, $v_{\text{max}}$ | $\rho$, grid size $L$ | $p_{\text{rand}}$, $\rho$ |

---

## How to Read the Plots

### Fundamental Diagram — Flow $J$ vs Density $\rho$ (Section 4)

The x-axis is the car density $\rho \in [0, 1]$ (fraction of cells occupied); the y-axis is the average flow $J$ (cars passing a fixed point per timestep).

- **Low density (free flow)**: Few cars, little interaction. Flow rises roughly linearly — adding more cars directly adds more throughput.
- **Critical density**: Flow reaches its maximum. The road is optimally loaded.
- **High density (congestion)**: Cars are too close together and must constantly brake. Flow falls back toward zero.

This hump shape is the hallmark of the NaSch model. Higher $v_\text{max}$ lifts the peak because faster-moving cars can carry more flow before the road saturates. The slope changes sharply at the critical density — a phase transition between free-flow and congested regimes.

### Flow vs Inflow Probability (Section 5)

The x-axis is the injection probability $p_\text{in}$ (0 = no cars ever enter, 1 = a car always tries to enter); the y-axis is the measured through-flow.

- **Low $p_\text{in}$**: The road is sparsely populated. Every car that enters eventually exits, so through-flow grows linearly with $p_\text{in}$.
- **High $p_\text{in}$**: The road reaches its capacity ceiling and through-flow saturates at a plateau regardless of how many more cars try to enter. Cars pile up near the entrance instead.
- Higher $v_\text{max}$ raises the saturation level because faster cars clear the road more quickly.

### Space-Time Diagrams (Section 6)

Each panel is a 200 × 500 pixel image: **rows = time** (top = $t = 0$, bottom = later), **columns = road position** (left = cell 0, right = cell $L-1$). A **black pixel** means the cell is occupied; **white** means empty.

#### Panel 1 — Free Flow ($p_\text{in} = 0.1$)

Sparse diagonal tracks slanting down to the right. Each continuous track is one car moving rightward at nearly constant speed (one cell per column shifts right per row). The tracks are widely spaced, reflecting low density. Cars rarely interact, and the occasional gap in a track is a single dawdling step.

#### Panel 2 — Congested ($p_\text{in} = 0.8$)

A dense, almost fully black image crossed by faint bands tilting to the **upper-right** (equivalently, propagating to the **left** over time). These are **stop-and-go waves**: clusters of cars forced to brake and re-accelerate, moving upstream through the traffic even though individual cars move downstream.

Crucially, there is no physical bottleneck — the waves arise entirely from the stochastic dawdling rule. One car randomly slows; the car behind must brake; that car's follower brakes further; the disturbance grows into a travelling jam. This is a hallmark of real motorway congestion.

#### Panel 3 — With 2 Traffic Lights ($p_\text{in} = 0.6$)

Two vertical **red dashed lines** mark the light positions (approximately cells 167 and 333). Several features are visible:

- **Dense vertical smears at the light positions**: Cars queue up during the red phase, producing a thick column of black pixels.
- **Regular gaps in those columns**: Each gap is a green phase — the queue discharges and the column temporarily clears.
- **Diagonal tracks between light positions**: Cars that cleared one light travel freely until the next, appearing as the familiar diagonal stripes from the free-flow panel.
- **Asymmetric queue lengths**: If the two lights are offset (green-wave mode), the queues at the two positions are shifted in time — one is building while the other is discharging — so the black smears do not align horizontally.

### Sensitivity Line Plots (sensitivity.ipynb, Sections 2–5)

Each plot holds all parameters fixed except one, which is swept along the x-axis; the y-axis is through-flow. The curves show where flow peaks and how steeply it falls away from the optimum. Two curves (synchronised vs offset) are overlaid in sections that compare coordination modes.

### 2D Heatmap (sensitivity.ipynb, Section 6)

A grid of colour-coded cells: x-axis = $T_\text{cycle}$, y-axis = $N$. Warmer colours (yellow/red) mean higher through-flow; cooler colours (purple/blue) mean lower flow. The **red star** marks the parameter combination that achieved the highest flow. Two side-by-side heatmaps compare synchronised and offset coordination; comparing them directly shows where the green-wave strategy gains the most.

### Stochastic BML — Overlaid Fundamental Diagrams (stochastic_bml.ipynb, Section 3)

Four curves share the same axes: the leftmost ($p=0$) is the sharp BML drop. As $p_{\text{rand}}$
increases the curve shifts left and the drop becomes a gradual slope. Vertical dotted lines mark
each curve's $\rho_c$ (estimated from the susceptibility peak). The separation between dotted
lines quantifies how much dawdling shifts the critical density.

### Stochastic BML — Susceptibility Peaks (stochastic_bml.ipynb, Section 4)

Two panels side by side for $L = 128$. **Left**: order parameter $\phi$ vs $\rho$ — curves
overlap at low density and diverge near the transition. **Right**: $-\mathrm{d}\phi/\mathrm{d}\rho$
(susceptibility) — each curve has a peak whose x-position is $\rho_c$ and whose height measures
the sharpness of the transition. A tall, narrow peak ($p=0$) indicates a near-discontinuous
transition; a short, broad peak (large $p$) indicates a smooth crossover. Vertical dashed lines
mark each $\rho_c$.

### Stochastic BML — Flow vs $p_{\text{rand}}$ (stochastic_bml.ipynb, Section 7)

Three curves at fixed densities sweep $p_{\text{rand}}$ along the x-axis. The **free-flow
curve** ($\rho=0.10$) falls monotonically — every extra unit of dawdling directly wastes
capacity. The **critical curve** ($\rho \approx \rho_c$) may be non-monotone: a small
$p_{\text{rand}}$ can break proto-jam clusters before they lock in, briefly helping flow
before the dawdling cost dominates. The **dense curve** ($\rho=0.45$) may rise initially
as stochastic unfreezing allows escape from near-gridlock configurations that the deterministic
model would never leave.

### Summary Bar Chart (sensitivity.ipynb, Section 7)

Five scenarios at fixed $p_\text{in}$ plotted as side-by-side bars: no lights, worst-case lights, optimal synchronised, and optimal offset. The bar heights make the cost of traffic lights and the benefit of green-wave coordination immediately legible.

---

## Key Results

1. **Fundamental diagram**: Flow peaks at intermediate density and matches the classic NaSch hump. Higher $v_\text{max}$ raises the capacity (peak flow) because faster cars clear gaps more quickly, allowing more cars to be on the road productively.

2. **Open boundary**: Through-flow saturates once inflow exceeds road capacity. Beyond the saturation point, extra cars entering only lengthen the queue — they do not increase throughput.

3. **Stop-and-go waves**: Visible in space-time diagrams as backward-propagating diagonal bands under congested conditions. They arise from a cascade: one driver randomly dawdles → the car behind must brake harder → its follower brakes further → a jam nucleates and propagates upstream even as individual cars move forward.

4. **Traffic lights always reduce maximum through-flow** relative to no lights, because every red phase is a mandatory halt that limits how many cars can cross a given point per unit time. The reduction is minimised by:
   - Choosing $T_\text{cycle}$ that matches car travel time between lights: too short → cars spend a disproportionate fraction of time stopped; too long → large queues build during red and take many green steps to drain.
   - Using **offset (green wave)** coordination: cars arrive at the next light precisely when it turns green, so they pass without stopping. Synchronised lights, by contrast, create a wall of red that stops every car in the platoon simultaneously.
   - Keeping $N$ small: each additional light is an independent bottleneck, and their effects compound.

5. **Optimal configuration** (for default parameters): found via heatmap sweep of $(N, T_\text{cycle})$ — marked with a star in Section 6 of `sensitivity.ipynb`. The offset mode consistently outperforms synchronised mode at moderate inflow.

6. **Stochastic BML bridges NaSch and BML**: injecting NaSch-style dawdling ($p_{\text{rand}} = 0.3$) into the 2-D BML grid softens the sharp first-order-like phase transition into a gradual crossover. The critical density $\rho_c$ shifts downward with $p_{\text{rand}}$, and configurations that are permanent absorbing gridlocks in the deterministic model can be partially escaped — gridlock becomes probabilistic rather than irreversible. Free-flow throughput is always reduced by dawdling. Traffic lights compound both effects: the combined model shows approximately additive flow penalties from stochasticity and signal control.

---

## How to Run

1. Open a terminal in the project root.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the desired notebook:
   ```bash
   jupyter notebook notebooks/nagel_schreckenberg.ipynb       # NaSch model + traffic lights
   jupyter notebook notebooks/nagel_sensitivity.ipynb         # NaSch sensitivity analysis
   jupyter notebook notebooks/biham_middleton_levine.ipynb    # BML model
   jupyter notebook notebooks/stochastic_bml.ipynb            # Stochastic BML (NaSch + BML)
   jupyter notebook notebooks/simulations/stochastic_bml_animation.ipynb  # Stochastic BML animations
   ```
4. Run all cells top-to-bottom (**Kernel → Restart & Run All**).

Typical runtimes (all parameter sweeps):
- `nagel_schreckenberg.ipynb` — ~3–5 min (most expensive: Section 8 heatmap)
- `nagel_sensitivity.ipynb` — ~5–8 min
- `biham_middleton_levine.ipynb` — ~5–10 min (most expensive: Section 8, L=256 sweep)
- `stochastic_bml.ipynb` — ~8–12 min (most expensive: Sections 4 and 7, L=128 sweeps across p_rand values)
- `stochastic_bml_animation.ipynb` — ~3–5 min (Animation 2 pre-computes 120 grids with warmup; Animations 1 and 3 are real-time)

---

## Parameters

All key parameters are defined as constants at the top of Section 1:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `L` | 500 | Road length (cells) |
| `V_MAX` | 5 | Default maximum velocity (cells/step) |
| `P_RAND` | 0.3 | Dawdling probability — 30% chance of random deceleration per car per step |
| `T_WARMUP` | 500 | Transient steps discarded before measurement; ensures the road has reached statistical steady state |
| `T_MEASURE` | 1000 | Steps averaged for flow — longer gives lower variance |

---

## References

- [17] M. Rickert, K. Nagel, M. Schreckenberg, A. Latour. *Two lane traffic simulations using cellular automata.* Physica A, 231:534–550, 1996.
- [18] M. Schreckenberg, A. Schadschneider, K. Nagel, N. Ito. *Discrete stochastic-models for traffic flow.* Phys. Rev. E, 51:2939–2949, 1995.
- [6] D. Helbing. *Traffic and related self-driven many-particle systems.* Rev. Mod. Phys., 73:1067–1141, 2001.
- O. Biham, A. A. Middleton, D. Levine. *Self-organization and a dynamical transition in traffic-flow models.* Phys. Rev. A, 46:R6124–R6127, 1992.
