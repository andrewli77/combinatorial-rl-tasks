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
from flat_model import ACModel

# Parse arguments

parser = argparse.ArgumentParser()

## General parameters
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=30,
                    help="number of updates between two saves (default: 30, 0 means no saving)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")
parser.add_argument("--wandb", action="store_true", default=False,
                    help="Log the experiment with weights & biases")
parser.add_argument("--checkpoint-dir", default=None)

## Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=10,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=1600,
                    help="batch size for PPO (default: 1600)")
parser.add_argument("--frames-per-proc", type=int, default=2000,
                    help="number of frames per process before update (default: 5 for A2C and 2000 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.0003,
                    help="learning rate (default: 0.0003)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.003,
                    help="entropy term coefficient (default: 0.003)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--num-training-tasks", type=int, default=100000,
                    help="Number of unique seeds the agent is allowed to train on.")
parser.add_argument("--hidden-size", type=int, default=185)
parser.add_argument("--distributional-value", action="store_true", default=False)

args = parser.parse_args()

args.mem = args.recurrence > 1
assert(args.recurrence == 1)

# Set run dir

date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"{args.env}_flat_{args.num_training_tasks}-tasks_seed{args.seed}"

storage_dir = "storage" if args.checkpoint_dir is None else args.checkpoint_dir
model_dir = utils.get_model_dir(default_model_name, storage_dir)
in_model_dir = None if args.model is None else utils.get_model_dir(args.model, "")
# Load loggers and Tensorboard writer

txt_logger = utils.get_txt_logger(model_dir)
csv_file, csv_logger = utils.get_csv_logger(model_dir)
#tb_writer = tensorboardX.SummaryWriter(model_dir)
if not args.wandb:
    os.environ['WANDB_MODE'] = 'disabled'
wandb.init(project="hrl")
wandb.run.name = default_model_name
wandb.run.save()
config = wandb.config
config.algo = "PPO"
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
    envs.append(make_train_env(args.env, hier=False, num_training_tasks=args.num_training_tasks, rng_seed=args.seed+10000*i))
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
acmodel = ACModel(obs_space, envs[0].action_space, args.hidden_size, args.distributional_value)
print("Low parameters:", sum(p.numel() for p in acmodel.parameters() if p.requires_grad))

if "model_state" in status:
    acmodel.load_state_dict(status["model_state"])

acmodel.to(device)
txt_logger.info("Model loaded\n")
txt_logger.info("{}\n".format(acmodel))
# Load algo

algo = torch_ac.algos.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda, args.distributional_value,
                        args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                        args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)

if "optimizer_state" in status:
    algo.optimizer.load_state_dict(status["optimizer_state"])
txt_logger.info("Optimizer loaded\n")

# Train model

num_frames = status["num_frames"]
update = status["update"]
start_time = time.time()

while num_frames < args.frames:
    # Update model parameters

    update_start_time = time.time()
    exps, logs1 = algo.collect_experiences()
    logs2 = algo.update_parameters(exps)
    logs = {**logs1, **logs2}
    update_end_time = time.time()

    num_frames += logs["num_frames"]
    update += 1

    # Print logs

    if update % args.log_interval == 0:
        fps = logs["num_frames"]/(update_end_time - update_start_time)
        duration = int(time.time() - start_time)
        return_per_episode = utils.synthesize(logs["return_per_episode"])['mean']
        rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])['mean']
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])['mean']
        header = ["update", "frames", "FPS", "duration", "return", "reshaped_return", "num_frames"]
        data = [update, num_frames, fps, duration, return_per_episode, rreturn_per_episode, num_frames_per_episode]
        header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
        data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

        txt_logger.info(
            "U {} | F {:06} | FPS {:04.0f} | D {} | return: {:.2f} | reward: {:.2f} | F: {:.1f} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
            .format(*data))

        if args.distributional_value:
            header += ["value_std"]
            data += [logs["value_std"]]

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
                    "model_state": acmodel.state_dict(),
                    "optimizer_state": algo.optimizer.state_dict()}
        if hasattr(preprocess_obss, "vocab"):
            status["vocab"] = preprocess_obss.vocab.vocab
        utils.save_status(status, model_dir)
        txt_logger.info("Status saved")
