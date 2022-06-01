import numpy as np
import gym

import copy
from TSP_env import * 
from zone_envs.ZoneEnvBase import ZoneEnvBase, zone


visited = zone.Yellow
unvisited = zone.Cyan
class TSPHardEnv(TSPEnv):
    def __init__(self, config):
        config = copy.deepcopy(config)
        self.zone_colours = config.pop("zones_colours", None)
        self.time_saved_reward = config.pop("time_saved_reward", 0.01)
        self.num_cities = config.pop("num_cities")

        # We are going to use the decorated getter for this value so stop Engine from
        # creating the attribute
        if 'reward_goal' in self.DEFAULT:
            self.DEFAULT.pop("reward_goal")
        config.update({'continue_goal': False})
        self.zone_types = [visited, unvisited]
        self.high_only_keys = ['remaining'] + [f'zones_lidar_{i}' for i in range(self.num_cities)]
        ZoneEnvBase.__init__(self, zones=[zone(zone_colour) for zone_colour in self.zone_colours], config=config)

    def reset(self):
        self.zones = [zone(zone_colour) for zone_colour in self.zone_colours]
        return ZoneEnvBase.reset(self)
