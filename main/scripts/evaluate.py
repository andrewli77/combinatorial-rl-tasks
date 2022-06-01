import argparse
import time
import numpy
import torch
import os
import pickle

import utils
from envs.make_env import make_test_env, make_fixed_env

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--distributional-value", action="store_true", default=False)
args = parser.parse_args()


## Evaluate on 100 maps, 5 runs per map 
n_maps = 100
n_runs_per_map = 5
seed = 0

utils.seed(seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load agent

model_dir = utils.get_model_dir(args.model)
env = make_fixed_env(args.env, hier=False, seed=seed, env_seed=0) # Dummy env to build the agent
agent = utils.Agent(env.observation_space, env.action_space, model_dir, distributional_value=True,
                    device=device)
print("Agent loaded\n")

# Recording results

pkl_path = os.path.join(args.model, "results-%s.pkl"%(args.env))
record_returns = []

for env_seed in range(1000000, 1000000+n_maps):
    env = make_fixed_env(args.env, hier=False, seed=seed, env_seed=env_seed)

    returns_this_run = []
    print ("Env Seed", env_seed)
    for run in range(n_runs_per_map):
        total_reward = 0
        eps_len = 0

        obs = env.reset()

        while True:
            action = agent.get_action(obs)
            obs, reward, done, info = env.step(action)
            eps_len += 1
            total_reward += reward

            if done:
                if 'goal_met' in info and info['goal_met']:
                    print("Success! --- Total reward: %.3f --- Eps len: %d" %(total_reward, eps_len))

                else:
                    print("Fail! --- Total reward: %.3f --- Eps len: %d" %(total_reward, eps_len))

                returns_this_run.append(total_reward)
                break

    record_returns.append(returns_this_run)

    pkl_file = open(pkl_path, "wb")
    pickle.dump({"return": record_returns}, pkl_file)
    pkl_file.close()

