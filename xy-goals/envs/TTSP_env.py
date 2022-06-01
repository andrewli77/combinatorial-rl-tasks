import numpy as np
from gym.envs.registration import register

from TSP_env import *
"""
    Same as TSPEnv but adds a random timeout to each city (zone). The agent has to visit the city within that timeout
    or it will lose. The timeouts are sampled from a beta distribution to avoid having small timeout cities.

    @params:
        beta_a, beta_b: the a nd b parameters of the beta distribution.
"""
class TimedTSPEnv(TSPEnv):
    def __init__(self, config, beta_a=3, beta_b=1.5):
        super().__init__(config)
        self.beta_a = beta_a
        self.beta_b = beta_b
        self.max_steps = config["num_steps"]

    def reset_zone_times(self):
        rs = np.random.RandomState(self._seed)
        self.zone_max_steps = np.array([int(rs.beta(self.beta_a, self.beta_b) * self.max_steps) for _ in self.zones])

    @property
    def zone_times(self):
        _out = (self.zone_max_steps - self.steps) / self.max_steps
        _out[np.where(np.array(self.zones) == visited)] = 1.
        return _out

    def set_mocaps(self):
        super().set_mocaps()

        _zone_times = self.zone_times 
        for i, zone in enumerate(self.zones):
            if zone != visited:
                rgba = self._rgb[unvisited].copy()
                # The colour of unvisited (Cyan) is #00FFFF. The following lines gradually increase the value of the first two
                # digits while decreasing the last four turning the colour to Red #FF0000. The "type" of the city does not change
                # as a result of it turning to red, it is still considered "Unvisited".
                rgba[0] = 1 - _zone_times[i]
                rgba[1] = max(0, _zone_times[i])
                rgba[2] = max(0, _zone_times[i])
                body_id = self.sim.model.geom_name2id(f'zone{i}')
                self.sim.model.geom_rgba[body_id] = rgba

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if done:
            return obs, reward, done, info

        if (self.zone_times <= 0).any():
            self.done = True
            done = True

        return obs, reward, self.done, info

    def reset(self):
        self.steps = 0
        self.reset_zone_times()
        return super().reset()

    def build_zone_observation_space(self):
        for i, zone in enumerate(self.zones):
            # Dimensionality is 7 because:
            #    position is 2-dimensional
            #    colour (RGBA) is 4-dimensional and
            #    zone time is 1-dimensional
            self.obs_space_dict.update({f'zones_lidar_{i}': gym.spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32)})

    def obs_zones(self, obs):
        _zone_times = self.zone_times
        for i, zone in enumerate(self.zones):
            pos = self.data.get_body_xpos(f'zone{i}').copy()[:2] / 3.
            colour = self._rgb[zone]
            time_remaining = _zone_times[i]
            obs[f'zones_lidar_{i}'] = np.concatenate((pos, colour, [time_remaining]))
