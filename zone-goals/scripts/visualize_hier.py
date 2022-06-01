import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import utils
from envs.make_env import make_test_env
# from sequence.sequence_helper import *

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--episodes", type=int, default=10,
                    help="number of episodes to visualize")
parser.add_argument("--no-render", action="store_true", default=False)
parser.add_argument("--use-solver", action="store_true", default=False)

args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environment

env = make_test_env(args.env, hier=True, seed=args.seed)
env.reset()
print("Environment loaded\n")

# Load agent

model_dir = utils.get_model_dir(args.model)

agent = utils.HierAgent(env.observation_space, env.action_space, model_dir, env.unwrapped.num_cities, 2, device=device)
print("Agent loaded\n")

# Create a window to view the environment
if not args.no_render:
    env.render()

avg_reward = 0

for episode in range(args.episodes):
    total_reward = 0
    obs = env.reset()

    i = 0
    while True:
        if not args.no_render:
            env.render()

        # Update the skill every `skill_len` steps
        if env.goal_zone == None:
            if args.use_solver:
                goal = torch.tensor(env.solver_get_next_goal())
            else:
                available_goals = env.get_available_goals()
                goal = agent.get_hi_action(obs, available_goals)
            env.set_goal(goal.cpu().numpy())

        goal_xy = torch.tensor([env.get_goal()], dtype=torch.float32)

        lo_action = agent.get_lo_action(obs, goal_xy)[0]
        obs, reward, done, info = env.step(lo_action)
        total_reward += reward

        if reward != 0:
            print(reward)

        i += 1
        if done:
            if 'goal_met' in info and info['goal_met']:
                print("Success! --- Reward: %.3f" %(total_reward))
            else:
                print("Fail! --- Reward: %.3f" %(total_reward))
            avg_reward += total_reward
            break

print("Average reward:", avg_reward / args.episodes)

