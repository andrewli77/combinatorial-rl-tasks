import numpy as np
import gym

import copy

from zone_envs.ZoneEnvBase import ZoneEnvBase, zone


visited = zone.Yellow
unvisited = zone.Cyan
class TSPEnv(ZoneEnvBase):
    def __init__(self, config):
        config = copy.deepcopy(config)
        self.time_saved_reward = config.pop("time_saved_reward", 0.01)
        self.num_cities = config.pop("num_cities")

        # We are going to use the decorated getter for this value so stop Engine from
        # creating the attribute
        if 'reward_goal' in self.DEFAULT:
            self.DEFAULT.pop("reward_goal")
        config.update({'continue_goal': False})
        self.zone_types = [visited, unvisited]
        self.high_only_keys = ['remaining'] + [f'zones_lidar_{i}' for i in range(self.num_cities)]
        super().__init__(zones=[unvisited]*self.num_cities, config=config)

    def build_zone_observation_space(self):
        for i, zone in enumerate(self.zones):
            # Dimensionality is 6 because position is 2-dimensional, colour (RGBA) is 4-dimensional
            self.obs_space_dict.update({f'zones_lidar_{i}': gym.spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32)})

    def obs_zones(self, obs):
        for i, zone in enumerate(self.zones):
            pos = self.data.get_body_xpos(f'zone{i}').copy()[:2] / 3.
            colour = self._rgb[zone]
            obs[f'zones_lidar_{i}'] = np.concatenate((pos, colour))

    @property
    def reward_goal(self):
        return (self.num_steps - self.steps) * self.time_saved_reward

    def reward(self):
        return 1 if self.new_city_reached else 0
        #return len([z for z in self.zones if z == visited])

    def step(self, action):
        self.zones_dirty = True
        self.new_city_reached = False

        return super().step(action)

    ## Hack!! This is the only function in Engine.step() that is called after updating agent's position
    ## and before calling obs() functions. Here we check if one of the cities has been visited by
    ## the agent. If so we update the type of the zones to 'visited'.
    def set_mocaps(self):
        if not self.zones_dirty: return

        for h_index, h_pos in enumerate(self.zones_pos):
            if self.zones[h_index] != visited:
                h_dist = self.dist_xy(h_pos)
                if h_dist <= self.zones_size:
                    self.zones[h_index] = visited
                    # self.world_config_dict['geoms'][f'zone{h_index}']["rgba"] = self._rgb[visited]
                    body_id = self.sim.model.geom_name2id(f'zone{h_index}')
                    self.sim.model.geom_rgba[body_id] = self._rgb[visited]
                    self.new_city_reached = True
                    # We assume the agent to be in one zone at a time
                    break
        self.zones_dirty = False

    def goal_met(self):
        return len([z for z in self.zones if z != visited]) == 0

    def reset(self):
        self.zones = [unvisited] * self.num_cities

        return super().reset()
