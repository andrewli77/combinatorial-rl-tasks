import numpy as np
import enum
import gym

from safety_gym.envs.engine import *
from mujoco_py import MjViewer, MujocoException, const, MjRenderContextOffscreen

DEFAULT_WIDTH = 100
DEFAULT_HEIGHT = 100

GROUP_ZONE = 7

class zone(enum.Enum):
    JetBlack = 0
    White    = 1
    Blue     = 2
    Green    = 3
    Red      = 4
    Yellow   = 5
    Cyan     = 6
    Magenta  = 7

    def __lt__(self, sth):
        return self.value < sth.value

    def __str__(self):
        return self.name[0]

    def __repr__(self):
        return self.name


class ZoneEnvBase(Engine):
    """
    This environment is a modification of the Safety-Gym's environment.
    There is no "goal circle" but rather a collection of zones that the
    agent has to visit or to avoid in order to finish the task.
    """
    def __init__(self, zones:list, config:dict={}):
        walled = config.pop("walled")
        world_extent = 3
        self.DEFAULT.update({
            'floor_display_mode': True,
            'observe_sensors': False,
            'observe_xy': True,
            'observe_zones': False,
            'zones_num': 0,  # Number of hazards in an environment
            'zones_placements': None,  # Placements list for hazards (defaults to full extents)
            'zones_locations': [],  # Fixed locations to override placements
            'zones_keepout': 0.55,  # Radius of hazard keepout for placement
            'zones_size': 0.2,  # Radius of hazards
            'placements_extents': [-world_extent, -world_extent, world_extent, world_extent]
        })

        if (walled):
            walls = [(i/10, j) for i in range(int(-world_extent * 10),int(world_extent * 10 + 1),1) for j in [-world_extent, world_extent]]
            walls += [(i, j/10) for i in [-world_extent, world_extent] for j in range(int(-world_extent * 10), int(world_extent * 10 + 1),1)]
            self.DEFAULT.update({
                'walls_num': len(walls),  # Number of walls
                'walls_locations': walls,  # This should be used and length == walls_num
                'walls_size': 0.1,  # Should be fixed at fundamental size of the world
            })

        self.zones = zones
        if not self.zone_types:
            self.zone_types = list(set(zones))
        self.zone_types.sort()
        self._rgb = {
            zone.JetBlack: [0, 0, 0, 0.25],
            zone.Blue    : [0, 0, 1, 0.25],
            zone.Green   : [0, 1, 0, 0.25],
            zone.Cyan    : [0, 1, 1, 0.25],
            zone.Red     : [1, 0, 0, 0.25],
            zone.Magenta : [1, 0, 1, 0.25],
            zone.Yellow  : [1, 1, 0, 0.25],
            zone.White   : [1, 1, 1, 0.25]
        }
        self.zone_rgbs = np.array([self._rgb[haz] for haz in self.zones])

        parent_config = {
            'robot_base': 'xmls/car.xml',
            'task': 'none',
            'lidar_num_bins': 16,
            'observe_zones': True,
            'zones_num': len(zones),
            'num_steps': 1000
        }
        parent_config.update(config)
        super().__init__(parent_config)

    @property
    def zones_pos(self):
        ''' Helper to get the zones positions from layout '''
        return [self.data.get_body_xpos(f'zone{i}').copy() for i in range(self.zones_num)]

    def build_observation_space(self):
        super().build_observation_space()

        if self.observe_zones:
            self.build_zone_observation_space()

        if self.observe_xy:
            self.obs_space_dict.update({'robot_pos': gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)})
            self.obs_space_dict.update({'robot_dir': gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)})
            self.obs_space_dict.update({'robot_velp': gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)})
            self.obs_space_dict.update({'robot_velr': gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)})

        if self.observation_flatten:
            self.obs_flat_size = sum([np.prod(i.shape) for i in self.obs_space_dict.values()])
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_flat_size,), dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Dict(self.obs_space_dict)

    def build_zone_observation_space(self):
        for zone_type in self.zone_types:
            self.obs_space_dict.update({f'zones_lidar_{zone_type}': gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)})

    def build_placements_dict(self):
        super().build_placements_dict()

        if self.zones_num: #self.constrain_hazards:
            self.placements.update(self.placements_dict_from_object('zone'))

    def build_world_config(self):
        world_config = super().build_world_config()

        for i in range(self.zones_num):
            name = f'zone{i}'
            geom = {'name': name,
                    'size': [self.zones_size, 1e-2],#self.zones_size / 2],
                    'pos': np.r_[self.layout[name], 2e-2],#self.zones_size / 2 + 1e-2],
                    'rot': self.random_rot(),
                    'type': 'cylinder',
                    'contype': 0,
                    'conaffinity': 0,
                    'group': GROUP_ZONE,
                    'rgba': self.zone_rgbs[i] * [1, 1, 1, 1.]} #0.1]}  # transparent
            world_config['geoms'][name] = geom

        return world_config


    def obs(self):
        ''' Return the observation of our agent '''
        self.sim.forward()  # Needed to get sensordata correct
        obs = {}

        if self.observe_goal_dist:
            obs['goal_dist'] = np.array([np.exp(-self.dist_goal())])
        if self.observe_goal_comp:
            obs['goal_compass'] = self.obs_compass(self.goal_pos)
        if self.observe_goal_lidar:
            obs['goal_lidar'] = self.obs_lidar([self.goal_pos], GROUP_GOAL)
        if self.task == 'push':
            box_pos = self.box_pos
            if self.observe_box_comp:
                obs['box_compass'] = self.obs_compass(box_pos)
            if self.observe_box_lidar:
                obs['box_lidar'] = self.obs_lidar([box_pos], GROUP_BOX)
        if self.task == 'circle' and self.observe_circle:
            obs['circle_lidar'] = self.obs_lidar([self.goal_pos], GROUP_CIRCLE)
        if self.observe_freejoint:
            joint_id = self.model.joint_name2id('robot')
            joint_qposadr = self.model.jnt_qposadr[joint_id]
            assert joint_qposadr == 0  # Needs to be the first entry in qpos
            obs['freejoint'] = self.data.qpos[:7]
        if self.observe_com:
            obs['com'] = self.world.robot_com()
        if self.observe_sensors:
            # Sensors which can be read directly, without processing
            for sensor in self.sensors_obs:  # Explicitly listed sensors
                obs[sensor] = self.world.get_sensor(sensor)
            for sensor in self.robot.hinge_vel_names:
                obs[sensor] = self.world.get_sensor(sensor)
            for sensor in self.robot.ballangvel_names:
                obs[sensor] = self.world.get_sensor(sensor)
            # Process angular position sensors
            if self.sensors_angle_components:
                for sensor in self.robot.hinge_pos_names:
                    theta = float(self.world.get_sensor(sensor))  # Ensure not 1D, 1-element array
                    obs[sensor] = np.array([np.sin(theta), np.cos(theta)])
                for sensor in self.robot.ballquat_names:
                    quat = self.world.get_sensor(sensor)
                    obs[sensor] = quat2mat(quat)
            else:  # Otherwise read sensors directly
                for sensor in self.robot.hinge_pos_names:
                    obs[sensor] = self.world.get_sensor(sensor)
                for sensor in self.robot.ballquat_names:
                    obs[sensor] = self.world.get_sensor(sensor)
        if self.observe_remaining:
            obs['remaining'] = np.array([1. - self.steps / self.num_steps])
            assert 0.0 <= obs['remaining'][0] <= 1.0, 'bad remaining {}'.format(obs['remaining'])
        if self.walls_num and self.observe_walls:
            obs['walls_lidar'] = self.obs_lidar(self.walls_pos, GROUP_WALL)
        if self.observe_hazards:
            obs['hazards_lidar'] = self.obs_lidar(self.hazards_pos, GROUP_HAZARD)
        if self.observe_vases:
            obs['vases_lidar'] = self.obs_lidar(self.vases_pos, GROUP_VASE)
        if self.gremlins_num and self.observe_gremlins:
            obs['gremlins_lidar'] = self.obs_lidar(self.gremlins_obj_pos, GROUP_GREMLIN)
        if self.pillars_num and self.observe_pillars:
            obs['pillars_lidar'] = self.obs_lidar(self.pillars_pos, GROUP_PILLAR)
        if self.buttons_num and self.observe_buttons:
            # Buttons observation is zero while buttons are resetting
            if self.buttons_timer == 0:
                obs['buttons_lidar'] = self.obs_lidar(self.buttons_pos, GROUP_BUTTON)
            else:
                obs['buttons_lidar'] = np.zeros(self.lidar_num_bins)
        if self.observe_qpos:
            obs['qpos'] = self.data.qpos.copy()
        if self.observe_qvel:
            obs['qvel'] = self.data.qvel.copy()
        if self.observe_ctrl:
            obs['ctrl'] = self.data.ctrl.copy()
        if self.observe_vision:
            obs['vision'] = self.obs_vision()
        if self.observe_zones:
            self.obs_zones(obs)
        if self.observe_xy:
            obs['robot_pos'] = self.world.robot_pos()[:2] / 3.
            robot_quat = self.data.get_body_xquat('robot').astype(np.float32)
            obs['robot_dir'] = np.array([robot_quat[0]**2 - robot_quat[3]**2, 2 * robot_quat[0] * robot_quat[3]])
            obs['robot_velp'] = self.world.robot_vel()[:2] / 1.5
            obs['robot_velr'] = np.array([self.data.get_body_xvelr('robot').copy()[2]]) / 3.
        if self.observation_flatten:
            flat_obs = np.zeros(self.obs_flat_size)
            offset = 0
            for k in sorted(self.obs_space_dict.keys()):
                k_size = np.prod(obs[k].shape)
                flat_obs[offset:offset + k_size] = obs[k].flat
                offset += k_size
            obs = flat_obs
    
        assert self.observation_space.contains(obs), f'Bad obs {obs} {self.observation_space}'
        return obs

    def obs_zones(self, obs):
        for zone_type in self.zone_types:
            ind = [i for i, z in enumerate(self.zones) if (self.zones[i] == zone_type)]
            pos_in_type = list(np.array(self.zones_pos)[ind])
            obs[f'zones_lidar_{zone_type}'] = self.obs_lidar(pos_in_type, GROUP_ZONE)

    def render(self,
               mode='human',
               camera_id=None,
               width=DEFAULT_WIDTH,
               height=DEFAULT_HEIGHT
               ):
        ''' Render the environment to the screen '''

        if self.viewer is None or mode!=self._old_render_mode:
            # Set camera if specified
            if mode == 'human':
                self.viewer = MjViewer(self.sim)
                self.viewer.cam.fixedcamid = -1
                self.viewer.cam.type = const.CAMERA_FREE
            else:
                self.viewer = MjRenderContextOffscreen(self.sim)
                self.viewer._hide_overlay = True
                self.viewer.cam.fixedcamid = camera_id #self.model.camera_name2id(mode)
                self.viewer.cam.type = const.CAMERA_FIXED
            self.viewer.render_swap_callback = self.render_swap_callback
            # Turn all the geom groups on
            self.viewer.vopt.geomgroup[:] = 1
            self._old_render_mode = mode
        self.viewer.update_sim(self.sim)

        if camera_id is not None:
            # Update camera if desired
            self.viewer.cam.fixedcamid = camera_id

        # Lidar markers
        if self.render_lidar_markers:
            offset = self.render_lidar_offset_init  # Height offset for successive lidar indicators
            if 'box_lidar' in self.obs_space_dict or 'box_compass' in self.obs_space_dict:
                if 'box_lidar' in self.obs_space_dict:
                    self.render_lidar([self.box_pos], COLOR_BOX, offset, GROUP_BOX)
                if 'box_compass' in self.obs_space_dict:
                    self.render_compass(self.box_pos, COLOR_BOX, offset)
                offset += self.render_lidar_offset_delta
            if 'goal_lidar' in self.obs_space_dict or 'goal_compass' in self.obs_space_dict:
                if 'goal_lidar' in self.obs_space_dict:
                    self.render_lidar([self.goal_pos], COLOR_GOAL, offset, GROUP_GOAL)
                if 'goal_compass' in self.obs_space_dict:
                    self.render_compass(self.goal_pos, COLOR_GOAL, offset)
                offset += self.render_lidar_offset_delta
            if 'buttons_lidar' in self.obs_space_dict:
                self.render_lidar(self.buttons_pos, COLOR_BUTTON, offset, GROUP_BUTTON)
                offset += self.render_lidar_offset_delta
            if 'circle_lidar' in self.obs_space_dict:
                self.render_lidar([ORIGIN_COORDINATES], COLOR_CIRCLE, offset, GROUP_CIRCLE)
                offset += self.render_lidar_offset_delta
            if 'walls_lidar' in self.obs_space_dict:
                self.render_lidar(self.walls_pos, COLOR_WALL, offset, GROUP_WALL)
                offset += self.render_lidar_offset_delta
            if 'hazards_lidar' in self.obs_space_dict:
                self.render_lidar(self.hazards_pos, COLOR_HAZARD, offset, GROUP_HAZARD)
                offset += self.render_lidar_offset_delta
            if 'pillars_lidar' in self.obs_space_dict:
                self.render_lidar(self.pillars_pos, COLOR_PILLAR, offset, GROUP_PILLAR)
                offset += self.render_lidar_offset_delta
            if 'gremlins_lidar' in self.obs_space_dict:
                self.render_lidar(self.gremlins_obj_pos, COLOR_GREMLIN, offset, GROUP_GREMLIN)
                offset += self.render_lidar_offset_delta
            if 'vases_lidar' in self.obs_space_dict:
                self.render_lidar(self.vases_pos, COLOR_VASE, offset, GROUP_VASE)
                offset += self.render_lidar_offset_delta

            for zone_type in self.zone_types:
                if f'zones_lidar_{zone_type}' in self.obs_space_dict:
                    ind = [i for i, z in enumerate(self.zones) if (self.zones[i] == zone_type)]
                    pos_in_type = list(np.array(self.zones_pos)[ind])

                    self.render_lidar(pos_in_type, np.array([self._rgb[zone_type]]), offset, GROUP_ZONE)
                    offset += self.render_lidar_offset_delta

        # Add goal marker
        if self.task == 'button':
            self.render_area(self.goal_pos, self.buttons_size * 2, COLOR_BUTTON, 'goal', alpha=0.1)

        # Add indicator for nonzero cost
        if self._cost.get('cost', 0) > 0:
            self.render_sphere(self.world.robot_pos(), 0.25, COLOR_RED, alpha=.5)

        # Draw vision pixels
        if self.observe_vision and self.vision_render:
            vision = self.obs_vision()
            vision = np.array(vision * 255, dtype='uint8')
            vision = Image.fromarray(vision).resize(self.vision_render_size)
            vision = np.array(vision, dtype='uint8')
            self.save_obs_vision = vision

        if mode=='human':
            self.viewer.render()
        elif mode=='rgb_array':
            self.viewer.render(width, height)
            data = self.viewer.read_pixels(width, height, depth=False)
            self.viewer._markers[:] = []
            self.viewer._overlay.clear()
            return data[::-1, :, :]




