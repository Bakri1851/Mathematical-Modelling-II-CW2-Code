"""Space-time recording and plotting helpers for two-lane NaSch sims.

Three plotting helpers (velocity-coloured, identity-coloured, type-coloured)
all share a 2-panel (lane 0 top, lane 1 bottom) layout with time increasing
downward. Empty cells are rendered white via masked arrays.

Functions
---------
record_space_time(sim, T_record, T_warmup=None) -> dict
plot_space_time_velocity(history, ...) -> Figure
plot_space_time_identity(history, ..., highlight_ids=None) -> Figure
plot_space_time_type(history, ...) -> Figure
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


def record_space_time(sim, T_record, T_warmup=None):
    """Snapshot road/car_ids/v_max_arr per timestep into stacked arrays.

    Parameters
    ----------
    sim : TwoLaneNaSch (or subclass) — caller seeds RNG before construction.
    T_record : int — number of timesteps to snapshot post-warmup.
    T_warmup : int or None — optional warmup before recording starts.

    Returns
    -------
    dict with keys:
        velocity_history    (T_record, 2, L) int   — sim.road snapshots
        car_id_history      (T_record, 2, L) int
        v_max_history       (T_record, 2, L) int
        light_state_history (T_record, n_lights) bool — only if sim has lights
        light_pos           (n_lights,) int        — only if sim has lights
        parameters          dict — every relevant sim parameter
    """
    if T_warmup is not None and T_warmup > 0:
        for _ in range(T_warmup):
            sim.step()

    L = sim.L
    velocity_history = np.full((T_record, 2, L), -1, dtype=int)
    car_id_history = np.full((T_record, 2, L), -1, dtype=int)
    v_max_history = np.full((T_record, 2, L), -1, dtype=int)

    has_lights = hasattr(sim, "light_pos") and hasattr(sim, "light_states")
    if has_lights:
        n_lights = len(sim.light_pos)
        light_state_history = np.zeros((T_record, n_lights), dtype=bool)
    else:
        light_state_history = None

    for t in range(T_record):
        sim.step()
        velocity_history[t] = sim.road
        car_id_history[t] = sim.car_ids
        v_max_history[t] = sim.v_max_arr
        if has_lights:
            states = sim.light_states()
            for i, pos in enumerate(sim.light_pos):
                light_state_history[t, i] = (states[int(pos)] == "red")

    parameters = dict(
        L=L,
        n_lanes=sim.n_lanes,
        v_max=sim.v_max,
        p_rand=sim.p_rand,
        p_chg=sim.p_chg,
        boundary=sim.boundary,
        p_in=sim.p_in,
        v_max_slow=sim.v_max_slow,
        v_max_fast=sim.v_max_fast,
        f_slow=sim.f_slow,
        T_warmup=T_warmup if T_warmup is not None else 0,
        T_record=T_record,
    )
    if has_lights:
        parameters.update(
            N=sim.N,
            T_cycle=sim.T_cycle,
            T_green=sim.T_green,
            phase_offsets=np.asarray(sim.phase_offsets).copy(),
        )

    out = dict(
        velocity_history=velocity_history,
        car_id_history=car_id_history,
        v_max_history=v_max_history,
        parameters=parameters,
    )
    if has_lights:
        out["light_state_history"] = light_state_history
        out["light_pos"] = np.asarray(sim.light_pos).copy()
    return out


def _crop(history_field, time_range, cell_range):
    """Slice (T, 2, L) arrays by time and cell ranges. Returns sliced array
    plus the (x0, x1, t0, t1) extent for imshow."""
    T, _, L = history_field.shape
    t0, t1 = (0, T) if time_range is None else time_range
    x0, x1 = (0, L) if cell_range is None else cell_range
    return history_field[t0:t1, :, x0:x1], (x0, x1, t1, t0)


def _add_caption(fig, caption):
    if caption:
        # Below the figure, doesn't fight constrained_layout.
        fig.text(0.5, -0.02, caption, ha="center", va="top",
                 fontsize=9, wrap=True)


def _overlay_lights(ax, light_pos, light_state_history, time_range, cell_range):
    """Stripe each light position with red/green bands per timestep."""
    if light_state_history is None or light_pos is None:
        return
    T = light_state_history.shape[0]
    t0, t1 = (0, T) if time_range is None else time_range
    x0, x1 = ax.get_xlim()
    for li, pos in enumerate(light_pos):
        if cell_range is not None and not (cell_range[0] <= pos < cell_range[1]):
            continue
        col = light_state_history[t0:t1, li]  # bool: True = red
        # render as a thin coloured strip using fill_betweenx per state run
        states = np.where(col, "red", "green")
        # Run-length encode for efficiency
        starts = [0]
        for i in range(1, len(states)):
            if states[i] != states[i - 1]:
                starts.append(i)
        starts.append(len(states))
        for k in range(len(starts) - 1):
            s, e = starts[k], starts[k + 1]
            colour = "red" if col[s] else "green"
            ax.fill_betweenx(
                [t0 + s, t0 + e],
                pos - 0.4, pos + 0.4,
                color=colour, alpha=0.4, linewidth=0,
            )


def plot_space_time_velocity(history, *, cell_range=None, time_range=None,
                              caption=None):
    """Two-panel velocity-coloured space-time. Colour = v / v_max^(i)."""
    vel, extent = _crop(history["velocity_history"], time_range, cell_range)
    vmax_arr, _ = _crop(history["v_max_history"], time_range, cell_range)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                              constrained_layout=True)
    # RdYlGn (not reversed) → 0=red (stopped), 1=green (at ceiling).
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("white")

    im = None
    for lane in (0, 1):
        v = vel[:, lane, :].astype(float)
        vm = vmax_arr[:, lane, :].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = np.where(vm > 0, v / np.maximum(vm, 1), np.nan)
        norm = np.where(vel[:, lane, :] < 0, np.nan, norm)
        masked = np.ma.masked_invalid(norm)
        im = axes[lane].imshow(
            masked, aspect="auto", origin="upper", extent=extent,
            cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
        )
        axes[lane].set_title(f"Lane {lane}", fontsize=11)
        axes[lane].set_ylabel("Time (timestep)")

        if "light_state_history" in history:
            _overlay_lights(
                axes[lane], history.get("light_pos"),
                history["light_state_history"], time_range, cell_range,
            )

    axes[1].set_xlabel("Cell position")
    cbar = fig.colorbar(im, ax=axes, orientation="vertical", shrink=0.8,
                         label="v / v_max^(i)")
    cbar.ax.text(0.5, 1.02, "1 = at indiv. ceiling",
                 transform=cbar.ax.transAxes, ha="center", fontsize=8)
    cbar.ax.text(0.5, -0.05, "0 = stopped",
                 transform=cbar.ax.transAxes, ha="center", fontsize=8)
    _add_caption(fig, caption)
    return fig


def plot_space_time_identity(history, *, highlight_ids=None,
                              cell_range=None, time_range=None,
                              caption=None):
    """Two-panel identity-coloured space-time. Each car_id maps to a stable
    colour via car_id % 20 → tab20. Same colour persists through lane-changes."""
    ids, extent = _crop(history["car_id_history"], time_range, cell_range)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                              constrained_layout=True)
    base_cmap = plt.get_cmap("tab20")
    n_colors = 20

    for lane in (0, 1):
        lane_ids = ids[:, lane, :]
        # Build an RGBA image directly so we can grey-out non-highlighted cars.
        rgba = np.ones(lane_ids.shape + (4,))  # white default
        rgba[..., 3] = 0.0  # transparent for empties
        car_mask = lane_ids >= 0
        if highlight_ids is None:
            colours = base_cmap((lane_ids % n_colors) / (n_colors - 1))
            rgba[car_mask] = colours[car_mask]
        else:
            highlight_set = set(int(i) for i in highlight_ids)
            highlight_arr = np.array([cid in highlight_set
                                       for cid in lane_ids.ravel()]).reshape(lane_ids.shape)
            # Non-highlighted cars: light grey
            non_hl = car_mask & ~highlight_arr
            rgba[non_hl] = (0.85, 0.85, 0.85, 0.5)
            # Highlighted cars: saturated tab20 colour
            colours = base_cmap((lane_ids % n_colors) / (n_colors - 1))
            hl_mask = car_mask & highlight_arr
            rgba[hl_mask] = colours[hl_mask]

        # imshow with explicit RGBA needs 3D array
        axes[lane].imshow(
            rgba, aspect="auto", origin="upper", extent=extent,
            interpolation="nearest",
        )
        axes[lane].set_title(f"Lane {lane}", fontsize=11)
        axes[lane].set_ylabel("Time (timestep)")

        if "light_state_history" in history:
            _overlay_lights(
                axes[lane], history.get("light_pos"),
                history["light_state_history"], time_range, cell_range,
            )

    axes[1].set_xlabel("Cell position")
    _add_caption(fig, caption)
    return fig


def plot_space_time_type(history, *, cell_range=None, time_range=None,
                          caption=None):
    """Two-panel type-coloured space-time. Blue = fast, red = slow, white empty."""
    vmax_arr, extent = _crop(history["v_max_history"], time_range, cell_range)
    p = history["parameters"]
    v_fast = p["v_max_fast"]
    v_slow = p["v_max_slow"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                              constrained_layout=True)
    # 0 = empty (white), 1 = fast (blue), 2 = slow (red)
    cmap = ListedColormap(["white", "#1f77b4", "#d62728"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    for lane in (0, 1):
        vm = vmax_arr[:, lane, :]
        coded = np.zeros_like(vm)
        coded[vm == v_fast] = 1
        coded[vm == v_slow] = 2
        # If v_fast == v_slow (homogeneous), treat all cars as fast.
        if v_fast == v_slow:
            coded[vm > 0] = 1
        axes[lane].imshow(
            coded, aspect="auto", origin="upper", extent=extent,
            cmap=cmap, norm=norm, interpolation="nearest",
        )
        axes[lane].set_title(f"Lane {lane}", fontsize=11)
        axes[lane].set_ylabel("Time (timestep)")

        if "light_state_history" in history:
            _overlay_lights(
                axes[lane], history.get("light_pos"),
                history["light_state_history"], time_range, cell_range,
            )

    axes[1].set_xlabel("Cell position")
    # Manual legend
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#1f77b4", label=f"fast (v_max={v_fast})"),
        Patch(facecolor="#d62728", label=f"slow (v_max={v_slow})"),
    ]
    fig.legend(handles=legend_elems, loc="upper right",
                bbox_to_anchor=(0.98, 0.98), fontsize=9)
    _add_caption(fig, caption)
    return fig


def find_lane_changers(car_id_history, max_ids=5):
    """Return up to max_ids car_ids that appeared in different lanes between
    consecutive frames during the recording window."""
    T = car_id_history.shape[0]
    changers = set()
    for t in range(1, T):
        # ids in lane 0 at t but in lane 1 at t-1, and vice versa
        for src, dst in [(0, 1), (1, 0)]:
            cur = set(int(c) for c in car_id_history[t, src] if c >= 0)
            prev = set(int(c) for c in car_id_history[t - 1, dst] if c >= 0)
            changers |= (cur & prev)
            if len(changers) >= max_ids * 4:
                break
    return sorted(changers)[:max_ids]
