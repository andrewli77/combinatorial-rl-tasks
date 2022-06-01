import argparse
import time
import numpy
import torch

import utils
from envs.make_env import make_test_env, make_fixed_env

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--env-seed", type=int, default=None)
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--episodes", type=int, default=10,
                    help="number of episodes to visualize")
parser.add_argument("--no_render", action="store_true", default=False)

args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environment
if args.env_seed is not None:
    env = make_fixed_env(args.env, hier=False, seed=args.seed, env_seed=args.env_seed)
else:
    env = make_test_env(args.env, hier=False, seed=args.seed)
env.reset()
print("Environment loaded\n")

# Load agent

model_dir = utils.get_model_dir(args.model)

agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                    device=device)
print("Agent loaded\n")

# Create a window to view the environment
if not args.no_render:
    env.render()
avg_reward = []
for episode in range(args.episodes):
    total_reward = 0
    eps_len = 0
    obs = env.reset()

    i = 0
    while True:
        if not args.no_render:
            env.render()

        action = agent.get_action(obs)
        obs, reward, done, info = env.step(action)
        eps_len += 1
        total_reward += reward * 0.99 ** i 

        if reward != 0:
            print(reward)

        i += 1

        if done:
            if 'goal_met' in info and info['goal_met']:
                print("Success! --- Total reward: %.3f --- Eps len: %d" %(total_reward, eps_len))
            else:
                print("Fail! --- Total reward: %.3f --- Eps len: %d" %(total_reward, eps_len))

            avg_reward.append(total_reward)
            break

print("Average reward:", numpy.mean(avg_reward), "Std:", numpy.std(avg_reward))

