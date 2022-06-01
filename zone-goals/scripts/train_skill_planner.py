import argparse
import time
import datetime
import torch
import torch_ac
import wandb
import sys
import os

import utils
from envs.make_env import make_train_env

from hier_policy_value_models import HighPolicyValueModel, LoPolicyValueModel
from env_model import getHiEnvEncoder
from inverse_model import InverseModel

# Parse arguments
parser = argparse.ArgumentParser()

## General parameters
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--pretrained-policy", default=None,
                    help="name of the pretrained model")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=30,
                    help="number of updates between two saves (default: 30, 0 means no saving)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=2*10**7,
                    help="number of frames of training (default: 2e7)")
parser.add_argument("--wandb", action="store_true", default=False,
                    help="Log the experiment with weights & biases")
parser.add_argument("--checkpoint-dir", default=None)

## Parameters for low (skill) policy optimization
parser.add_argument("--epochs", type=int, default=10,
                    help="number of epochs for PPO (default: 10)")
parser.add_argument("--batch-size", type=int, default=1600,
                    help="batch size for PPO (default: 200)")
parser.add_argument("--frames-per-proc", type=int, default=2000,
                    help="number of frames per process before update (default: 5 for A2C and 2000 for PPO)")
parser.add_argument("--lr", type=float, default=0.0003,
                    help="learning rate (default: 0.0003)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.003,
                    help="entropy term coefficient (default: 0.003)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")

## Parameters for high-level policy optimization
parser.add_argument("--hi-epochs", type=int, default=5)
parser.add_argument("--hi-batch-size", type=int, default=80)
parser.add_argument("--hi-lr", type=float, default=0.0003)
parser.add_argument("--hi-entropy-coef", type=float, default=0.1)
parser.add_argument("--hi-value-coef", type=float, default=0.5)

## Parameters for both
parser.add_argument("--hidden-size", type=int, default=128)
parser.add_argument("--max-grad-norm", type=float, default=10,
                    help="maximum norm of gradient (default: 2)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--num-training-tasks", type=int, default=100000,
                    help="Number of unique seeds the agent is allowed to train on.")
args = parser.parse_args()

# Set run dir
default_model_name = f"{args.env}_hier_{args.num_training_tasks}-tasks_seed{args.seed}"

storage_dir = "storage" if args.checkpoint_dir is None else args.checkpoint_dir
model_dir = utils.get_model_dir(default_model_name, storage_dir)
in_model_dir = None if args.model is None else utils.get_model_dir(args.model, "")

# Load loggers and Tensorboard writer
txt_logger = utils.get_txt_logger(model_dir)
csv_file, csv_logger = utils.get_csv_logger(model_dir)

if not args.wandb:
    os.environ['WANDB_MODE'] = 'disabled'
wandb.init(project="hrl")
wandb.run.name = default_model_name
wandb.run.save()
config = wandb.config
config.algo = "Hier-Planner"

# Log command and all script arguments
txt_logger.info("{}\n".format(" ".join(sys.argv)))
txt_logger.info("{}\n".format(args))

# Set seed for all randomness sources
utils.seed(args.seed)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
txt_logger.info(f"Device: {device}\n")

# Load environments
envs = []
for i in range(args.procs):
    envs.append(make_train_env(args.env, hier=True, num_training_tasks=args.num_training_tasks, rng_seed=args.seed+10000*i))

txt_logger.info("Environments loaded\n")

# Load training status
if in_model_dir:
    status = utils.get_status(in_model_dir)
else:
    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
txt_logger.info("Training status loaded\n")

# Load observations preprocessor
obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
if "vocab" in status:
    preprocess_obss.vocab.load_vocab(status["vocab"])
txt_logger.info("Observations preprocessor loaded")

# Load model
hi_policy_value_net = HighPolicyValueModel(obs_space, envs[0].unwrapped.num_cities, args.hidden_size)
lo_policy_value_net = LoPolicyValueModel(obs_space, envs[0].action_space, envs[0].unwrapped.goal_dim, args.hidden_size)

print("High parameters:", sum(p.numel() for p in hi_policy_value_net.parameters() if p.requires_grad))
print("Low parameters:", sum(p.numel() for p in lo_policy_value_net.parameters() if p.requires_grad))

if "hi_model_state" in status:
    hi_policy_value_net.load_state_dict(status["hi_model_state"])
if "lo_model_state" in status:
    lo_policy_value_net.load_state_dict(status["lo_model_state"])

# Load a pretrained policy 
if args.pretrained_policy is not None and args.model is None:
    pt_model_dir = utils.get_model_dir(args.pretrained_policy, "storage-good")
    pt_status = utils.get_status(pt_model_dir)
    lo_policy_value_net.load_state_dict(pt_status["lo_model_state"])

hi_policy_value_net.to(device)
lo_policy_value_net.to(device)

txt_logger.info("Model loaded\n")
txt_logger.info("{}\n".format(hi_policy_value_net))
txt_logger.info("{}\n".format(lo_policy_value_net))


# Load algo
policy_algo = torch_ac.algos.HierPolicyAlgo(envs, hi_policy_value_net, lo_policy_value_net, device, preprocess_obss,
                        args.epochs, args.batch_size, args.frames_per_proc, args.lr, args.gae_lambda, args.entropy_coef, args.discount, args.value_loss_coef, args.clip_eps,
                        args.hi_epochs, args.hi_batch_size, args.hi_entropy_coef, args.hi_value_coef, args.hi_lr,
                        args.max_grad_norm, args.optim_eps
                )

if "lo_optimizer_state" in status:
    policy_algo.lo_optimizer.load_state_dict(status["lo_optimizer_state"])
if "hi_optimizer_state" in status:
    policy_algo.hi_optimizer.load_state_dict(status["hi_optimizer_state"])
txt_logger.info("Optimizer loaded\n")

# Train model

num_frames = status["num_frames"]
update = status["update"]
start_time = time.time()

while num_frames < args.frames:
    # Update model parameters
    print(num_frames)
    time_1 = time.time()
    exps_policy, logs1 = policy_algo.collect_experiences()
    time_2 = time.time()
    logs2 = policy_algo.update_parameters(exps_policy)
    time_3 = time.time()

    logs = {**logs1, **logs2}

    num_frames += logs["num_frames"]
    update += 1

    # Print logs

    if update % args.log_interval == 0:
        policy_collect_time = time_2 - time_1
        policy_update_time = time_3 - time_2

        fps = logs["num_frames"] / (time_3 - time_1)

        return_per_episode = utils.synthesize(logs["return_per_episode"])['mean']
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])['mean']
        header = ["frames", "return", "num_frames", "FPS", "policy_collect_time", "policy_update_time"]
        data = [num_frames, return_per_episode, num_frames_per_episode, fps, policy_collect_time, policy_update_time]

        header += ["hi_entropy", "hi_value", "hi_policy_loss", "hi_value_loss", "hi_grad_norm"]
        data += [logs["hi_entropy"], logs["hi_value"], logs["hi_policy_loss"], logs["hi_value_loss"], logs["hi_grad_norm"]]

        header += ["lo_entropy", "lo_value", "lo_policy_loss", "lo_value_loss", "lo_grad_norm"] 
        data += [logs["lo_entropy"], logs["lo_value"], logs["lo_policy_loss"], logs["lo_value_loss"], logs["lo_grad_norm"]] 

        if status["num_frames"] == 0:
            csv_logger.writerow(header)
        csv_logger.writerow(data)
        csv_file.flush()
        for field, value in zip(header, data):
            #tb_writer.add_scalar(field, value, num_frames)
            wandb.log({field: value})
    # Save status
    if args.save_interval > 0 and update % args.save_interval == 0:
        status = {"num_frames": num_frames, "update": update,
                    "hi_model_state": hi_policy_value_net.state_dict(),
                    "lo_model_state": lo_policy_value_net.state_dict(),
                    "lo_optimizer_state": policy_algo.lo_optimizer.state_dict(),
                    "hi_optimizer_state": policy_algo.hi_optimizer.state_dict(),
                }
        if hasattr(preprocess_obss, "vocab"):
            status["vocab"] = preprocess_obss.vocab.vocab
        utils.save_status(status, model_dir)
        txt_logger.info("Status saved")
