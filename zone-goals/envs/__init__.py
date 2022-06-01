import envs.make_env, envs.push_env, envs.goal_env
from gym.envs.registration import register




config_point = {
    'robot_base': 'xmls/point.xml',
    'num_cities': 15,
    'walled': False,
    'observe_remaining': True,
    'observation_flatten': False,
    'num_steps': 2000
}

config_point_easy = {
    'robot_base': 'xmls/point.xml',
    'num_cities': 5,
    'walled': False,
    'observe_remaining': True,
    'observation_flatten': False,
    'num_steps': 1000
}

config_car = {
    'robot_base': 'xmls/car.xml',
    'num_cities': 15,
    'walled': False,
    'observe_remaining': True,
    'observation_flatten': False,
    'num_steps': 2000
}

config_doggo = {
    'robot_base': 'xmls/doggo.xml',
    'num_cities': 15,
    'walled': False,
    'observe_remaining': True,
    'observation_flatten': False,
    'num_steps': 2000
}

config_point_colour = {
    'robot_base': 'xmls/point.xml',
    'num_cities': 6,
    'walled': False,
    'observe_remaining': True,
    'observation_flatten': False,
    'num_steps': 2000
}

zone_locations_1 = [(-2.6, -1.6), (-0., -0.5), (1., 0.5), (1.8, 1.5), (2.6, 2.6)]

config_zone_fixed_1 = {
    'robot_base': 'xmls/point.xml',
    'num_cities': 15,
    'walled': False,
    'observe_remaining': True,
    'observation_flatten': False,
    'num_steps': 1000,
    'zones_num': 15,
    'zones_locations': zone_locations_1,
    'zones_colours': [6] * 5 + [5] * 10,
    'robot_locations': [(-0.9, -0.9)],
    'robot_rot': -1
}

zone_locations_2 = [(-2.6, -2.6), (-2, -1.6),] +\
                    [(2, 1),]
config_zone_fixed_2 = {
    'robot_base': 'xmls/point.xml',
    'num_cities': 15,
    'walled': False,
    'observe_remaining': True,
    'observation_flatten': False,
    'num_steps': 300,
    'zones_num': 15,
    'zones_locations': zone_locations_2,
    'zones_colours': [6] * 3 + [5] * 12,
    'robot_locations': [(0.8, 0.8)]
}


#======== TSP Envs ================
#---TSP-v0: 15 cities, 2000 step episodes
# Reward of 1 for each new city visited, plus bonus reward for reaching all cities (based on time remaining)
register(id='PointTSP-v0',
         entry_point='envs.TSP_env:TSPEnv',
         kwargs={'config': config_point})
#---TSP-v1: 5 cities, 1000 step episodes
# Reward of 1 for each new city visited, plus bonus reward for reaching all cities (based on time remaining)
register(id='PointTSP-v1',
         entry_point='envs.TSP_env:TSPEnv',
         kwargs={'config': config_point_easy})
#---TSP-v2: Order of cities is given by TSP solver.
# Dense (L2-distance-based) reward towards next unvisited city. The order of cities is part of the observation (the i-th city in the ordering is associated with a scalar observation of 2^{-i+1})
register(id='PointTSP-v2',
         entry_point='envs.TSP_order_env:TSPOrderEnv',
         kwargs={'config': config_point})
#---TSP-v2.1: Order of cities is given by TSP solver.
# Reward is same as original TSP reward
register(id='PointTSP-v21',
         entry_point='envs.TSP_order_test_env:TSPOrderTestEnv',
         kwargs={'config': config_point})
#---TSP-v3: Generates dense rewards towards a goal city. The goal city is requested when `info['need_next_goal']` is True and stays constant until the goal city is reached.
# Eligible goal cities (i.e. unvisited ones) are provided via `info['available_goals']. A new goal city can be set via the `set_goal` method. 
register(id='PointTSP-v3',
         entry_point='envs.TSP_next_city_env:TSPNextCityEnv',
         kwargs={'config': config_point})

#--TSP-v4: Hard instance of TSP 1
register(id='PointTSP-v4',
        entry_point='envs.TSP_hard_env:TSPHardEnv',
        kwargs={'config': config_zone_fixed_1})

#--TSP-v5: Hard instance of TSP 2
register(id='PointTSP-v5',
        entry_point='envs.TSP_hard_env:TSPHardEnv',
        kwargs={'config': config_zone_fixed_2})

register(id='CarTSP-v0',
         entry_point='envs.TSP_env:TSPEnv',
         kwargs={'config': config_car})

register(id='DoggoTSP-v0',
         entry_point='envs.TSP_env:TSPEnv',
         kwargs={'config': config_doggo})


# Timed TSP Envs
register(id='PointTTSP-v0',
         entry_point='envs.TTSP_env:TimedTSPEnv',
         kwargs={'config': config_point})

register(id='PointTTSP-v1',
         entry_point='envs.TTSP_env:TimedTSPEnv',
         kwargs={'config': config_point_easy})
#---TTSP-v3: Generates dense rewards towards a goal city. The goal city is requested when `info['need_next_goal']` is True and stays constant until the goal city is reached.
# Eligible goal cities (i.e. unvisited ones) are provided via `info['available_goals']. A new goal city can be set via the `set_goal` method. 
register(id='PointTTSP-v3',
         entry_point='envs.TTSP_next_city_env:TimedTSPNextCityEnv',
         kwargs={'config': config_point})

# Colour Match Envs
register(id='ColourMatch-v0',
         entry_point='envs.colour_match_env:ColourMatchEnv',
         kwargs={'config': config_point_colour})
register(id='ColourMatch-v2',
         entry_point='envs.colour_match_solver_env:ColourMatchSolverEnv',
         kwargs={'config': config_point_colour})
register(id='ColourMatch-v3',
         entry_point='envs.colour_match_next_city_env:ColourMatchNextCityEnv',
         kwargs={'config': config_point_colour})
