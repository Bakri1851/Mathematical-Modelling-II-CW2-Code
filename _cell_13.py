class TwoLaneNaSchWithLights(TwoLaneNaSch):
    """Two-lane NaSch with N evenly-spaced lights spanning both lanes.

    Light timing is delegated to a private `TrafficCAWithLights` instance
    (imported from the notebook via `_nasch_nb`) used as a schedule oracle:
    the single-lane code is the authoritative source for the
    `(N, T_cycle, T_green, offset)` → red-columns mapping, and this class
    does not re-implement it.
    """

    def __init__(
        self,
        L=500,
        n_lanes=2,
        v_max=5,
        p_rand=0.3,
        p_chg=1.0,
        boundary="open",
        n_cars=None,
        p_in=0.5,
        N=2,
        T_cycle=30,
        T_green=None,
        offset=False,
    ):
        super().__init__(
            L=L,
            n_lanes=n_lanes,
            v_max=v_max,
            p_rand=p_rand,
            p_chg=p_chg,
            boundary=boundary,
            n_cars=n_cars,
            p_in=p_in,
        )
        self._light_oracle = TrafficCAWithLights(
            L=L, N=N, T_cycle=T_cycle, T_green=T_green, offset=offset
        )
        self.N = N
        self.T_cycle = T_cycle
        self.T_green = self._light_oracle.T_green
        self.offset = offset
        self.light_pos = self._light_oracle.light_pos
        self.phase_offsets = self._light_oracle.phase_offsets

    def _red_light_positions(self):
        self._light_oracle.time = self.time
        return self._light_oracle._red_light_positions()

    def light_states(self):
        self._light_oracle.time = self.time
        return self._light_oracle.light_states()