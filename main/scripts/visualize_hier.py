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
parser.add_argument("--n-skills", type=int, default=5,
                    help="number of discrete skills the high-level policy can execute.")
parser.add_argument("--execute-skill", type=int, default=None)
parser.add_argument("--skill-len", type=int, default=200)
parser.add_argument("--no-render", action="store_true", default=False)
parser.add_argument("--planner", type=int, default=0)

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

agent = utils.HierAgent(env.observation_space, env.action_space, model_dir,
                    device=device, n_skills=args.n_skills, skill_len=args.skill_len)
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
        if i % args.skill_len == 0:
            # print(agent.high_value(obs))
            #agent.examine_next_states(obs)

            # Skill comes from argument (execute a fixed skill)
            if args.execute_skill is not None:
                skill = torch.tensor([args.execute_skill])

            # Skill comes from planner
            elif args.planner > 0:
                skill = agent.planner(obs, max_depth=args.planner)
                skill = torch.tensor([skill])

            # Skill comes from high-level policy
            else:
                skill = agent.get_hi_action(obs)
            print(skill)
        lo_action = agent.get_lo_action(obs, skill)[0]
        obs, reward, done, info = env.step(lo_action)
        total_reward += reward

        i += 1
        if done:
            if 'goal_met' in info and info['goal_met']:
                print("Success! --- Reward: %.3f" %(total_reward))
            else:
                print("Fail! --- Reward: %.3f" %(total_reward))
            avg_reward += total_reward
            break

print("Average reward:", avg_reward / args.episodes)

