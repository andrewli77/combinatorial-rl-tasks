import numpy as np
import gym

import copy

from zone_envs.ZoneEnvBase import ZoneEnvBase, zone
from src.utils.TSP_Solver import get_optim_route

visited = zone.Yellow
unvisited = zone.Cyan

# Observation includes the order the cities should be visited in, as provided by a TSP solver. 
# A scalar value of 2^{-i} for a city indicates it is the i-th city in the ordering 
# A shaped reward is given in info['shaped_reward']
class TSPOrderEnv(ZoneEnvBase):
    def __init__(self, config):
        config = copy.deepcopy(config)
        self.num_cities = config.pop("num_cities")
        self.time_saved_reward = config.pop("time_saved_reward", 0.01)

        # We are going to use the decorated getter for this value so stop Engine from
        # creating the attribute
        if 'reward_goal' in self.DEFAULT:
            self.DEFAULT.pop("reward_goal")
        config.update({'continue_goal': False})
        self.zone_types = [visited, unvisited]
        self.route = []
        self.high_only_keys = ['remaining'] + [f'zones_lidar_{i}' for i in range(self.num_cities)]

        super().__init__(zones=[unvisited]*self.num_cities, config=config)

    def build_zone_observation_space(self):
        for i, zone in enumerate(self.zones):
            # Dimensionality is 7 because position is 2-dimensional, colour (RGBA) is 4-dimensional, position in order is 1-dimensional
            self.obs_space_dict.update({f'zones_lidar_{i}': gym.spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32)})

    def obs_zones(self, obs):
        for i, zone in enumerate(self.zones):
            pos = self.data.get_body_xpos(f'zone{i}').copy()[:2] / 3.
            colour = self._rgb[zone]
            
            if i in self.route:
                order_val = np.power(0.5, self.route.index(i))
            else:
                order_val = 0.

            obs[f'zones_lidar_{i}'] = np.concatenate((pos, colour, [order_val]))

    def generate_route(self):
        self.route = list(get_optim_route(self))

    def dist_to_goal(self):
        if len(self.route) == 0:
            return 0
        next_pos = self.zones_pos[self.route[0]][:2]
        robot_pos = self.world.robot_pos()[:2]
        return np.sqrt(np.sum(np.square(next_pos - robot_pos)))

    @property
    def reward_goal(self):
        return (self.num_steps - self.steps) * self.time_saved_reward

    def reward(self):
        return 1 if self.new_city_reached else 0

    def shaped_reward(self):
        if self.new_city_reached:
            self.last_dist_to_goal = self.dist_to_goal()
            return 0
        else:
            _dist = self.dist_to_goal()
            _reward = self.last_dist_to_goal - _dist
            self.last_dist_to_goal = _dist
            return _reward

    def step(self, action):
        self.zones_dirty = True
        self.new_city_reached = False
        _o, _r, _d, _i = super().step(action)
        _i['shaped_reward'] = self.shaped_reward()
        return _o, _r, _d, _i

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
                    self.route.remove(h_index)
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
        init_obs = super().reset()
        self.generate_route()
        self.last_dist_to_goal = self.dist_to_goal()
        return init_obs
