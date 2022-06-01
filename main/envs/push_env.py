import safety_gym
from safety_gym.envs.engine import Engine
from safety_gym.random_agent import run_random

import numpy as np
import gym
from gym.envs.registration import register

# Obs_space: Dict(accelerometer:Box(-inf, inf, (3,), float32), velocimeter:Box(-inf, inf, (3,), float32), gyro:Box(-inf, inf, (3,), float32), magnetometer:Box(-inf, inf, (3,), float32), box_lidar:Box(0.0, 1.0, (16,), float32), goal_lidar:Box(0.0, 1.0, (16,), float32))
config_point = {
    'robot_base': 'xmls/point.xml',
    'task': 'push',
    'observation_flatten': False,
    'observe_goal_lidar': True,
    'observe_box_lidar': True,
    'continue_goal': False,
    'lidar_max_dist': 3,
    'lidar_num_bins': 16
}

# Obs_space: Dict(accelerometer:Box(-inf, inf, (3,), float32), velocimeter:Box(-inf, inf, (3,), float32), gyro:Box(-inf, inf, (3,), float32), magnetometer:Box(-inf, inf, (3,), float32), ballangvel_rear:Box(-inf, inf, (3,), float32), ballquat_rear:Box(-inf, inf, (3, 3), float32), box_lidar:Box(0.0, 1.0, (16,), float32), goal_lidar:Box(0.0, 1.0, (16,), float32))
config_car = {
    'robot_base': 'xmls/car.xml',
    'task': 'push',
    'observation_flatten': False,
    'observe_goal_lidar': True,
    'observe_box_lidar': True,
    'continue_goal': False,
    'lidar_max_dist': 3,
    'lidar_num_bins': 16
}

# Obs_space: Dict(accelerometer:Box(-inf, inf, (3,), float32), velocimeter:Box(-inf, inf, (3,), float32), gyro:Box(-inf, inf, (3,), float32), magnetometer:Box(-inf, inf, (3,), float32), jointvel_hip_1_z:Box(-inf, inf, (1,), float32), jointvel_hip_2_z:Box(-inf, inf, (1,), float32), jointvel_hip_3_z:Box(-inf, inf, (1,), float32), jointvel_hip_4_z:Box(-inf, inf, (1,), float32), jointvel_hip_1_y:Box(-inf, inf, (1,), float32), jointvel_hip_2_y:Box(-inf, inf, (1,), float32), jointvel_hip_3_y:Box(-inf, inf, (1,), float32), jointvel_hip_4_y:Box(-inf, inf, (1,), float32), jointvel_ankle_1:Box(-inf, inf, (1,), float32), jointvel_ankle_2:Box(-inf, inf, (1,), float32), jointvel_ankle_3:Box(-inf, inf, (1,), float32), jointvel_ankle_4:Box(-inf, inf, (1,), float32), jointpos_hip_1_z:Box(-inf, inf, (2,), float32), jointpos_hip_2_z:Box(-inf, inf, (2,), float32), jointpos_hip_3_z:Box(-inf, inf, (2,), float32), jointpos_hip_4_z:Box(-inf, inf, (2,), float32), jointpos_hip_1_y:Box(-inf, inf, (2,), float32), jointpos_hip_2_y:Box(-inf, inf, (2,), float32), jointpos_hip_3_y:Box(-inf, inf, (2,), float32), jointpos_hip_4_y:Box(-inf, inf, (2,), float32), jointpos_ankle_1:Box(-inf, inf, (2,), float32), jointpos_ankle_2:Box(-inf, inf, (2,), float32), jointpos_ankle_3:Box(-inf, inf, (2,), float32), jointpos_ankle_4:Box(-inf, inf, (2,), float32), box_lidar:Box(0.0, 1.0, (16,), float32), goal_lidar:Box(0.0, 1.0, (16,), float32))
config_doggo = {
    'robot_base': 'xmls/doggo.xml',
    'task': 'push',
    'observation_flatten': False,
    'observe_goal_lidar': True,
    'observe_box_lidar': True,
    'continue_goal': False,
    'lidar_max_dist': 3,
    'lidar_num_bins': 16   
}


config_car_flattened = {
    'robot_base': 'xmls/car.xml',
    'task': 'push',
    'observation_flatten': True,
    'observe_goal_lidar': True,
    'observe_box_lidar': True,
    'continue_goal': False,
    'lidar_max_dist': 3,
    'lidar_num_bins': 16
}

# Push envs
# Unflattened observations
register(id='PointPush-v0',
         entry_point='safety_gym.envs.mujoco:Engine',
         kwargs={'config': config_point})

register(id='CarPush-v0',
         entry_point='safety_gym.envs.mujoco:Engine',
         kwargs={'config': config_car})

register(id='DoggoPush-v0',
         entry_point='safety_gym.envs.mujoco:Engine',
         kwargs={'config': config_doggo})

# Flattened observations
register(id='CarPush-v1',
         entry_point='safety_gym.envs.mujoco:Engine',
         kwargs={'config': config_car_flattened})



