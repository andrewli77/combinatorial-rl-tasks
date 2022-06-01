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

args = parser.parse_args()

skill_len = 200

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
env = make_fixed_env(args.env, hier=True, seed=seed, env_seed=0) # Dummy env to build the agent
agent = utils.HierAgent(env.observation_space, env.action_space, model_dir,
                    device=device, n_skills=5, skill_len=skill_len)
print("Agent loaded\n")

# Recording results

pkl_path = os.path.join(args.model, "results-%s.pkl"%(args.env))
record_returns = []

for env_seed in range(1000000, 1000000+n_maps):
    env = make_fixed_env(args.env, hier=True, seed=seed, env_seed=env_seed)

    returns_this_run = []
    print ("Env Seed", env_seed)
    for run in range(n_runs_per_map):
        total_reward = 0
        i = 0

        obs = env.reset()

        while True:
            if i % skill_len == 0:
                skill = agent.get_hi_action(obs)

            lo_action = agent.get_lo_action(obs, skill)[0]
            obs, reward, done, info = env.step(lo_action)

            i += 1
            total_reward += reward

            if done:
                if 'goal_met' in info and info['goal_met']:
                    print("Success! --- Total reward: %.3f --- Eps len: %d" %(total_reward, i))

                else:
                    print("Fail! --- Total reward: %.3f --- Eps len: %d" %(total_reward, i))

                returns_this_run.append(total_reward)
                break

    record_returns.append(returns_this_run)

    pkl_file = open(pkl_path, "wb")
    pickle.dump({"return": record_returns}, pkl_file)
    pkl_file.close()

