import argparse
import time
import numpy as np
import glfw

import gym
import safety_gym
from gym import wrappers, logger
from envs.wrappers import PlayWrapper, FixedSeedsWrapper

from src.utils.TSP_Solver import get_optim_route

class RandomAgent(object):
    """This agent picks actions randomly"""
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, obs):
        return self.action_space.sample()

class PlayAgent(object):
    """
    This agent allows user to play with Safety's Point agent.

    Use the UP and DOWN arrows to move forward and back and
    use '<' and '>' to rotate the agent.
    """
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.prev_act = np.array([0, 0])
        self.last_obs = None

    def get_action(self, obs):
        # obs = obs["features"]

        key = self.env.key_pressed

        if(key == glfw.KEY_A):
            current = np.array([0, 0.4])
        elif(key == glfw.KEY_D):
            current = np.array([0, -0.4])
        elif(key == glfw.KEY_W):
            current = np.array([0.1, 0])
        elif(key == glfw.KEY_S):
            current = np.array([-0.1, 0])
        elif(key == -1): # This is glfw.RELEASE
            current = np.array([0, 0])
            self.prev_act = np.array([0, 0])
        else:
            current = np.array([0, 0])

        self.prev_act = np.clip(self.prev_act + current, -1, 1)

        return self.prev_act

def run_policy(agent, env, max_ep_len=None, num_episodes=100, render=True):
    env = wrappers.Monitor(env, directory="/tmp/zone_env.rec", force=True)
    # env.seed(1)
    show_optim = True

    o, r, d, ep_ret, ep_cost, ep_len, n = env.reset(), 0, False, 0, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        # env.show_text(message) # Uncomment to show a message on the top-right corner
        if show_optim:
            print(get_optim_route(env))
            print(f"optim rout: {[env.zones[city] for city in get_optim_route(env)]}")
            show_optim = False
        a = agent.get_action(o)
        a = np.clip(a, env.action_space.low, env.action_space.high)
        o, r, d, info = env.step(a)
        ep_ret += r
        ep_cost += info.get('cost', 0)
        ep_len += 1

        if r != 0:
            print("reward: %.3f" %(r))

        if d or (ep_len == max_ep_len):
            if(r):
                print("SUCCESS", ep_len)
            else:
                print("FAIL", ep_len)

            o, r, d, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0
            n += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', default='SafexpTest-v0', help='Select the environment to run')

    args = vars(parser.parse_args()) # make it a dictionary

    if "Doggo" in args["env_id"]:
        print("Doggo is not yet supported for manual control!!!")
        exit(1)

    env = gym.make(args["env_id"])
    env.num_steps = 10000000
    # env = PlayWrapper(MyFixedSeedsWrapper(env, min_seed=1, max_seed=2))
    env = PlayWrapper(env)

    agent = PlayAgent(env)

    run_policy(agent, env, max_ep_len=30000, num_episodes=1000)

