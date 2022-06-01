import numpy as np
import gym

import copy

from zone_envs.ZoneEnvBase import ZoneEnvBase, zone


colours = [zone.Blue, zone.Green, zone.Red]

class ColourMatchEnv(ZoneEnvBase):
    def __init__(self, config):
        config = copy.deepcopy(config)
        self.time_saved_reward = config.pop("time_saved_reward", 0.01)
        self.num_cities = config.pop("num_cities")
        self.max_cd = 150

        # We are going to use the decorated getter for this value so stop Engine from
        # creating the attribute
        if 'reward_goal' in self.DEFAULT:
            self.DEFAULT.pop("reward_goal")
        config.update({'continue_goal': False})
        self.zone_types = colours
        super().__init__(zones=[zone.Blue]*self.num_cities, config=config)

    def cycle_zones(self, h_index):
        if self.zones[h_index] == zone.Blue:
            self.zones[h_index] = zone.Green
        elif self.zones[h_index] == zone.Green:
            self.zones[h_index] = zone.Red
        elif self.zones[h_index] == zone.Red:
            self.zones[h_index] = zone.Blue

        self.zone_cooldowns[h_index] = self.max_cd

        return self.zones[h_index]

    def hamming_dist_to_goal(self):
        n_green = 0
        n_blue = 0
        n_red = 0

        for i in range(self.num_cities):
            if self.zones[i] == zone.Blue:
                n_blue += 1
            elif self.zones[i] == zone.Green:
                n_green += 1
            elif self.zones[i] == zone.Red:
                n_red += 1

        dist_to_blue = n_green * 2 + n_red
        dist_to_green = n_red * 2 + n_blue
        dist_to_red = n_blue * 2 + n_green

        return min(dist_to_blue, dist_to_green, dist_to_red)

    def reset_zones(self):
        _max_tries = 100
        for i in range(_max_tries):
            rs = np.random.RandomState(self._seed)

            self.zones = [rs.choice(colours) for i in range(self.num_cities)]
            self.zone_rgbs = np.array([self._rgb[haz] for haz in self.zones])
            self.zone_cooldowns = [0 for i in range(self.num_cities)]
            self.goal_dist = self.hamming_dist_to_goal()

            if self.goal_dist > 0:
                break

    def build_zone_observation_space(self):
        for i, zone in enumerate(self.zones):
            # Dimensionality is 7 because position is 2-dimensional, colour (RGBA) is 4-dimensional, cooldown is 1-dimensional
            self.obs_space_dict.update({f'zones_lidar_{i}': gym.spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32)})

    def obs_zones(self, obs):
        for i, zone in enumerate(self.zones):
            pos = self.data.get_body_xpos(f'zone{i}').copy()[:2] / 3.
            colour = self._rgb[zone]
            cooldown = np.array([np.float32(self.zone_cooldowns[i]) / self.max_cd])
            obs[f'zones_lidar_{i}'] = np.concatenate((pos, colour, cooldown))

    @property
    def reward_goal(self):
        return (self.num_steps - self.steps) * self.time_saved_reward

    def reward(self):
        if self.zones_changed:
            _new_dist = self.hamming_dist_to_goal()
            _reward = self.goal_dist - _new_dist
            self.goal_dist = _new_dist
            return _reward
        else:
            return 0

    def step(self, action):
        self.zones_dirty = True
        self.zones_changed = False
        for i in range(len(self.zone_cooldowns)):
            if self.zone_cooldowns[i] > 0:
                self.zone_cooldowns[i] -= 1
        return super().step(action)

    ## Hack!! This is the only function in Engine.step() that is called after updating agent's position
    ## and before calling obs() functions. Here we check if one of the cities has been visited by
    ## the agent. If so we update the type of the zones to 'visited'.
    def set_mocaps(self):
        if not self.zones_dirty: return

        for h_index, h_pos in enumerate(self.zones_pos):
            if self.zone_cooldowns[h_index] == 0:
                h_dist = self.dist_xy(h_pos)
                if h_dist <= self.zones_size:
                    new_colour = self.cycle_zones(h_index)
                    body_id = self.sim.model.geom_name2id(f'zone{h_index}')
                    self.sim.model.geom_rgba[body_id] = self._rgb[new_colour]
                    self.zones_changed = True

                    # We assume the agent to be in one zone at a time
                    break
        self.zones_dirty = False

    def goal_met(self):
        return self.goal_dist == 0

    def reset(self):
        self.reset_zones()
        return super().reset()
