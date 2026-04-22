"""Two-road intersection built on the single-lane NaSch model.

Option B, minimum viable version: two perpendicular single-lane
open-boundary NaSch roads share one crossing cell at ``L // 2``.
A single binary phase variable governs which road is green at each
timestep; the other is red. No lane-changing, no red-light
violations, no N x M generalisation.

Implementation note
-------------------
This module defines a private subclass ``_RoadWithExternalLight``
that overrides :meth:`nasch.TrafficCA._red_light_positions` to return
the crossing cell when its ``is_red`` flag is set. The flag is toggled
each step by :class:`IntersectionSystem` according to the global
phase. The base class ``TrafficCA`` is unchanged; this is the same
hook-override extension pattern used by
:class:`nasch.TrafficCAWithLights`.

Contents
--------
* ``IntersectionSystem`` -- two-road crossing with mutually-exclusive
  traffic lights, parameterised by cycle length ``T`` and green-split
  ``eta = T_g1 / T``.
"""

import numpy as np

from nasch import TrafficCA


class _RoadWithExternalLight(TrafficCA):
    """TrafficCA with a single externally-toggled red light.

    ``is_red`` is set by the enclosing :class:`IntersectionSystem`
    before each call to :meth:`step`. When True, the crossing cell is
    treated as an obstacle for this road's braking rule -- identical
    behaviour to a stationary car at that cell.
    """

    def __init__(self, *, crossing_cell, **kwargs):
        super().__init__(**kwargs)
        self.crossing_cell = crossing_cell
        self.is_red = False

    def _red_light_positions(self):
        if self.is_red:
            return np.array([self.crossing_cell], dtype=int)
        return np.array([], dtype=int)


class IntersectionSystem:
    """Two perpendicular single-lane roads sharing a crossing cell.

    Parameters
    ----------
    L        : road length (same for both roads)
    v_max    : velocity cap
    p_rand   : dawdling probability
    p_in_1   : inflow probability for road 1
    p_in_2   : inflow probability for road 2
    T        : total light cycle length (timesteps)
    eta      : green-time split for road 1; ``T_g1 = round(eta * T)``,
               ``T_g2 = T - T_g1``. Road 2 is green whenever road 1
               is red, and vice versa.
    seed     : optional; if provided, seeds ``numpy.random`` once at
               construction for reproducibility.

    Phase convention
    ----------------
    ``phase = time % T``. ``phase in [0, T_g1)`` -> road 1 green /
    road 2 red. ``phase in [T_g1, T)`` -> road 1 red / road 2 green.
    Mutual exclusion is structural (a single boolean is toggled).
    """

    def __init__(self, L, v_max, p_rand, p_in_1, p_in_2,
                 T, eta, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.L        = L
        self.T        = T
        self.eta      = eta
        self.T_g1     = int(round(eta * T))
        self.T_g2     = T - self.T_g1
        self.crossing = L // 2
        self.road1 = _RoadWithExternalLight(
            L=L, v_max=v_max, p_rand=p_rand, p_in=p_in_1,
            crossing_cell=self.crossing,
        )
        self.road2 = _RoadWithExternalLight(
            L=L, v_max=v_max, p_rand=p_rand, p_in=p_in_2,
            crossing_cell=self.crossing,
        )
        self.time = 0

    def _road1_is_green(self):
        return (self.time % self.T) < self.T_g1

    def _road2_is_green(self):
        return not self._road1_is_green()

    def step(self, record_flow=False):
        self.road1.is_red = not self._road1_is_green()
        self.road2.is_red = not self._road2_is_green()
        self.road1.step(record_flow=record_flow)
        self.road2.step(record_flow=record_flow)
        self.time += 1

    def warmup(self, T):
        for _ in range(T):
            self.step(record_flow=False)
        self.road1.outflow_count = 0
        self.road2.outflow_count = 0

    def run(self, T):
        """Run T measurement steps; return (J1, J2, J_total) in cars/step."""
        self.road1.outflow_count = 0
        self.road2.outflow_count = 0
        for _ in range(T):
            self.step(record_flow=True)
        J1 = self.road1.outflow_count / T
        J2 = self.road2.outflow_count / T
        return J1, J2, J1 + J2
