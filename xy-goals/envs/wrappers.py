import gym
import glfw
import numpy as np
from mujoco_py import MjViewer, const

## Use this to ensure you only train on a fixed set of seeds. 
## Each episode randomly samples a seed between [min_seed, max_seed]
## rng_seed: Make sure it's different for different processes, otherwise the parallel processes 
##           will sample the same sequence of tasks.
class FixedSeedsWrapper(gym.Wrapper):
    def __init__(self, env, min_seed, max_seed, rng_seed=0):
        super().__init__(env)
        self.env = env

        self.min_seed = min_seed
        self.max_seed = max_seed
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(seed=rng_seed)

    def reset(self):
        new_seed = self.rng.integers(low=self.min_seed, high=self.max_seed+1, size=1)[0]
        self.env.seed(new_seed)
        return self.env.reset()

## Allows you to call step on an env that is "done". 
## We use this to sync up all the envs so the high-level policies can execute on the same step. 
## step(): If the inner env is done, this method has the effect of a no-op. Returns an observation of all zeroes
##                  and zero reward with a done signal. 
class WaitWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.inner_done = False

    def step(self, action):
        if not self.inner_done:
            obs, rew, done, info = self.env.step(action)
            if done:
                self.inner_done = True
        else:
            obs = self.noop_obs() # zeroes observation
            rew = 0
            done = True
            info = {}        

        return obs, rew, done, info

    def noop_obs(self):
        return {
         'zone_obs': np.zeros(self.observation_space.spaces['zone_obs'].shape),
         'obs': np.zeros(self.observation_space.spaces['obs'].shape)}

    def reset(self):
        self.inner_done = False
        return self.env.reset() 
        
# class HierWrapper(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
        
#         if not hasattr(self.unwrapped, 'high_only_keys'):
#             raise RuntimeError('Environment must have high_only_keys as a property')
        
#         self.observation_space = self.split_hier_obs_space()

#     def step(self, action):
#         obs, rew, done, info = self.env.step(action)
#         obs = self.split_hier_obs(obs)
#         return obs, rew, done, info

#     def split_hier_obs(self, obs):
#         hi_obs = np.concatenate([obs[k].flatten() for k in obs.keys()])
#         lo_obs = np.concatenate([obs[k].flatten() for k in obs.keys() if k not in self.unwrapped.high_only_keys])
#         return {
#             'hi_obs': hi_obs,
#             'lo_obs': lo_obs
#         }

#     def split_hier_obs_space(self):
#         obs_sample = self.env.observation_space.sample()
#         obs_sample = self.split_hier_obs(obs_sample)
        
#         obs_space = gym.spaces.Dict({
#             'hi_obs': gym.spaces.Box(low=-np.inf,high=np.inf, shape=(obs_sample['hi_obs'].shape)),
#             'lo_obs': gym.spaces.Box(low=-np.inf,high=np.inf, shape=(obs_sample['lo_obs'].shape))
#         })

#         return obs_space

#     def reset(self):
#         return self.split_hier_obs(self.env.reset())

# # Handles observations in ZoneEnv with a variable number of zones.
# # obs['zone_obs']: shape (k, zone_obs_dim) where k is number of zones.
# # obs['hi_obs']: shape (hi_obs_dim)
# # obs['lo_obs']: shape (lo_obs_dim) 
# class ZoneHierWrapper(HierWrapper):
#     def __init__(self, env):
#         super().__init__(env)

#     def split_hier_obs(self, obs):
#         zone_obs = np.stack([obs[k].flatten() for k in obs.keys() if "zones_full_lidar" in k]) # A variable length tensor
#         hi_obs = np.concatenate([obs[k].flatten() for k in obs.keys() if "zones_full_lidar" not in k])
#         lo_obs = np.concatenate([obs[k].flatten() for k in obs.keys() if k not in self.unwrapped.high_only_keys and "zones_full_lidar" not in k])
#         return {
#             'zone_obs': zone_obs,
#             'hi_obs': hi_obs,
#             'lo_obs': lo_obs
#         }

#     def split_hier_obs_space(self):
#         obs_sample = self.env.observation_space.sample()
#         obs_sample = self.split_hier_obs(obs_sample)
        
#         obs_space = gym.spaces.Dict({
#             'zone_obs': gym.spaces.Box(low=-np.inf,high=np.inf, shape=(obs_sample['zone_obs'].shape)),
#             'hi_obs': gym.spaces.Box(low=-np.inf,high=np.inf, shape=(obs_sample['hi_obs'].shape)),
#             'lo_obs': gym.spaces.Box(low=-np.inf,high=np.inf, shape=(obs_sample['lo_obs'].shape))
#         })

#         return obs_space

# Handles observations in ZoneEnv with a variable number of zones.
# obs['zone_obs']: shape (k, zone_obs_dim) where k is number of zones.
# obs['obs']: shape (obs_dim)
class ZoneWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
        self.observation_space = self.split_zone_obs_space()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        obs = self.split_zone_obs(obs)
        return obs, rew, done, info

    def split_zone_obs(self, obs):
        zone_obs = np.stack([obs[k].flatten() for k in obs.keys() if "zones_lidar" in k])
        obs = np.concatenate([obs[k].flatten() for k in obs.keys() if "zones_lidar" not in k])
        return {
            'zone_obs': zone_obs,
            'obs': obs
        }

    def split_zone_obs_space(self):
        obs_sample = self.env.observation_space.sample()
        obs_sample = self.split_zone_obs(obs_sample)
        
        obs_space = gym.spaces.Dict({
            'zone_obs': gym.spaces.Box(low=-np.inf,high=np.inf, shape=(obs_sample['zone_obs'].shape)),
            'obs': gym.spaces.Box(low=-np.inf,high=np.inf, shape=(obs_sample['obs'].shape))
        })

        return obs_space

    def reset(self):
        return self.split_zone_obs(self.env.reset())


## Guarantees that the environment runs exactly `max_timeout` steps before terminating. If it finishes early,
## we return the last observation and 0 reward repeatedly until 10000 steps is done. 
class TimeoutWrapper(gym.Wrapper):
    def __init__(self, env, max_timeout=10000):
        super().__init__(env)
        self.max_timeout = max_timeout
        self.timer = 0
        self.inner_done = False
        self.last_obs = None

    def step(self, action):
        self.timer += 1
        if not self.inner_done:
            obs, rew, done, info = self.env.step(action)
            if done:
                self.inner_done = True
                self.last_obs = obs
            
            done = False
            info = {"timer": self.timer}
        else:
            obs = self.last_obs
            rew = 0
            done = False
            info = {"timer": self.timer}

        if self.timer == self.max_timeout:
            done = True

        return obs, rew, done, info

    def reset(self):
        self.timer = 0
        self.inner_done = False
        self.last_obs = None
        return self.env.reset() 


## A simple wrapper for testing SafetyGym-based envs. It uses the PlayViewer that listens to
## key_pressed events and passes the id of the pressed key as part of the observation to the agent.
## (used to control the agent via keyboard)

## Note: This should NOT be used for training!
class PlayWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.key_pressed = None

    # Shows a text on the upper right corner of the screen
    def show_text(self, text):
        self.env.viewer.show_text(text)

    def render(self, mode='human'):
        if self.env.viewer is None:
            self.env._old_render_mode = 'human'
            self.env.viewer = PlayWrapper.PlayViewer(self.env.sim)
            self.env.viewer.cam.fixedcamid = -1
            self.env.viewer.cam.type = const.CAMERA_FREE

            self.env.viewer.render_swap_callback = self.env.render_swap_callback
            # Turn all the geom groups on
            self.env.viewer.vopt.geomgroup[:] = 1
            self.env._old_render_mode = mode

        super().render()

    def wrap_obs(self, obs):
        if not self.env.viewer is None:
            self.key_pressed = self.env.viewer.consume_key()

        return obs

    def reset(self):
        obs = self.env.reset()

        return self.wrap_obs(obs)

    def step(self, action):
        next_obs, original_reward, env_done, info = self.env.step(action)

        return self.wrap_obs(next_obs), original_reward, env_done, info


    class PlayViewer(MjViewer):
        def __init__(self, sim):
            super().__init__(sim)
            self.key_pressed = None
            self.custom_text = None

            glfw.set_window_size(self.window, 840, 680)

        def show_text(self, text):
            self.custom_text = text

        def consume_key(self):
            ret = self.key_pressed
            self.key_pressed = None

            return ret

        def key_callback(self, window, key, scancode, action, mods):
            self.key_pressed = key
            if action == glfw.RELEASE:
                self.key_pressed = -1

            super().key_callback(window, key, scancode, action, mods)

        def _create_full_overlay(self):
            if (self.custom_text):
                self.add_overlay(const.GRID_TOPRIGHT, "Message", self.custom_text)


            step = round(self.sim.data.time / self.sim.model.opt.timestep)
            self.add_overlay(const.GRID_BOTTOMRIGHT, "Step", str(step))
            self.add_overlay(const.GRID_BOTTOMRIGHT, "timestep", "%.5f" % self.sim.model.opt.timestep)
            # self.add_overlay(const.GRID_BOTTOMRIGHT, "n_substeps", str(self.sim.nsubsteps))