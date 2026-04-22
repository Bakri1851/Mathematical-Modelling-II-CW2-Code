class TrafficCAWithLights(TrafficCA):
    """
    NaSch CA extended with N equidistant traffic lights.

    Parameters
    ----------
    N          : number of traffic lights
    T_cycle    : total switching cycle length (timesteps)
    T_green    : number of green timesteps per cycle (T_red = T_cycle - T_green)
    offset     : if True, consecutive lights are phase-shifted by T_cycle/N
    """

    def __init__(self, L=500, v_max=5, p_rand=0.3, p_in=0.5,
                 N=2, T_cycle=30, T_green=None, offset=False):
        super().__init__(L=L, v_max=v_max, p_rand=p_rand, p_in=p_in)
        self.N       = N
        self.T_cycle = T_cycle
        self.T_green = T_green if T_green is not None else T_cycle // 2
        self.offset  = offset

        # Light positions — evenly spaced
        self.light_pos = np.array(
            [L * (i + 1) // (N + 1) for i in range(N)], dtype=int
        )

        # Phase offsets per light
        if offset:
            self.phase_offsets = np.array(
                [int(i * T_cycle / N) for i in range(N)], dtype=int
            )
        else:
            self.phase_offsets = np.zeros(N, dtype=int)

    def _light_is_red(self, light_idx):
        """Return True if light `light_idx` is currently red."""
        phase = (self.time + self.phase_offsets[light_idx]) % self.T_cycle
        return phase >= self.T_green  # green phase first, then red

    def _red_light_positions(self):
        """Return positions of all currently red lights."""
        red = [self.light_pos[i] for i in range(self.N) if self._light_is_red(i)]
        return np.array(red, dtype=int)

    def light_states(self):
        """Return dict of {position: 'green'/'red'} for current timestep."""
        return {
            self.light_pos[i]: ('red' if self._light_is_red(i) else 'green')
            for i in range(self.N)
        }


print("TrafficCAWithLights class defined.")