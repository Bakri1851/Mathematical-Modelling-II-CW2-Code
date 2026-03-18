# Mathematical Modelling 2 — CW2: Traffic Flow

Stochastic cellular automaton model for single-lane one-way traffic flow, with and without equidistant traffic lights.

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

The randomisation step (dawdling) introduces stochasticity and is responsible for the spontaneous formation of **stop-and-go waves** — backward-propagating congestion bands observed in real traffic.

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

### `notebooks/traffic_flow_analysis.ipynb`

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

---

## Key Results

1. **Fundamental diagram**: Flow peaks at intermediate density and matches the classic NaSch hump. Higher $v_\text{max}$ raises the capacity (peak flow).

2. **Open boundary**: Through-flow saturates once inflow exceeds road capacity. Saturation level increases with $v_\text{max}$.

3. **Stop-and-go waves**: Visible in space-time diagrams as backward-propagating diagonal bands under congested conditions.

4. **Traffic lights** always reduce maximum through-flow relative to no lights, but the reduction is minimised by:
   - Choosing $T_\text{cycle}$ that matches car travel time between lights
   - Using offset (green wave) coordination
   - Not using too many lights for the road length

5. **Optimal configuration** (for default parameters): found via heatmap sweep of $(N, T_\text{cycle})$ — marked with a star in Section 8.

---

## How to Run

1. Open a terminal in the project root.
2. Install dependencies if needed:
   ```bash
   pip install numpy matplotlib jupyter
   ```
3. Launch the notebook:
   ```bash
   jupyter notebook notebooks/traffic_flow_analysis.ipynb
   ```
4. Run all cells top-to-bottom (**Kernel → Restart & Run All**).

Typical runtime: ~3–5 minutes for all parameter sweeps (most expensive: Section 8 heatmap).

---

## Parameters

All key parameters are defined as constants at the top of Section 1:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `L` | 500 | Road length (cells) |
| `V_MAX` | 5 | Default maximum velocity |
| `P_RAND` | 0.3 | Dawdling probability |
| `T_WARMUP` | 500 | Discard steps before measurement |
| `T_MEASURE` | 1000 | Steps averaged for flow |

---

## References

- [17] M. Rickert, K. Nagel, M. Schreckenberg, A. Latour. *Two lane traffic simulations using cellular automata.* Physica A, 231:534–550, 1996.
- [18] M. Schreckenberg, A. Schadschneider, K. Nagel, N. Ito. *Discrete stochastic-models for traffic flow.* Phys. Rev. E, 51:2939–2949, 1995.
- [6] D. Helbing. *Traffic and related self-driven many-particle systems.* Rev. Mod. Phys., 73:1067–1141, 2001.
