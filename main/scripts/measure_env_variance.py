import argparse
import time
import numpy
import torch
import pickle
import os

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
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")

args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load agent
env = make_fixed_env(args.env, hier=False, seed=args.seed, env_seed=0)
model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                    device=device)
print("Agent loaded\n")


rewards = numpy.zeros((20, 20, 2000)) # env_seed, episode #, trajectory
n_env_seeds = 20
n_episodes = 20


for env_seed in range(n_env_seeds):
    env = make_fixed_env(args.env, hier=False, seed=args.seed, env_seed=env_seed)
    env.reset()

    for episode in range(n_episodes):
        obs = env.reset()

        i = 0
        while True:
            action = agent.get_action(obs)
            obs, reward, done, info = env.step(action)
            rewards[env_seed][episode][i] = reward

            i += 1

            if done:
                if 'goal_met' in info and info['goal_met']:
                    print("Success!")
                else:
                    print("Fail!")
                break

pkl_path = os.path.join(args.model, "measure-variance.pkl")

pkl_file = open(pkl_path, "wb")
pickle.dump(rewards, pkl_file)
pkl_file.close()

