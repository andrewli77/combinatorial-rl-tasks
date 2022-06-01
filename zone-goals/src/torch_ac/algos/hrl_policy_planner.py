import torch
from envs.wrappers import WaitWrapper
from torch_ac.format import default_preprocess_obss
from torch_ac.torch_utils import DictList, ParallelEnv

import numpy as np

# PPO with a high level and low level policy. The high level policy only takes a step every `skill_len` steps.
class HierPolicyAlgo:
    """The base class for RL algorithms."""

    def __init__(self, envs, hi_policy_value_net, lo_policy_value_net, device, preprocess_obss,
            epochs, batch_size, num_frames_per_proc, lr, gae_lambda, entropy_coef, discount, value_loss_coef, clip_eps,
            hi_epochs, hi_batch_size, hi_entropy_coef, hi_value_coef, hi_lr,
            max_grad_norm, optim_eps
        ):

        # Store parameters
        self.device = device
        self.env = ParallelEnv(envs)
        self.hi_policy_value_net = hi_policy_value_net
        self.lo_policy_value_net = lo_policy_value_net
        self.preprocess_obss = preprocess_obss or default_preprocess_obss



        # Low policy (skill) optimization hyperparams
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_frames_per_proc = num_frames_per_proc
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.discount = discount
        self.value_loss_coef = value_loss_coef
        self.clip_eps = clip_eps

        # High policy optimization hyperparams
        self.hi_epochs = hi_epochs
        self.hi_batch_size = hi_batch_size
        self.hi_entropy_coef = hi_entropy_coef
        self.hi_value_coef = hi_value_coef
        self.hi_lr = hi_lr

        # Other hyperparams
        self.max_grad_norm = max_grad_norm
        self.optim_eps = optim_eps
        self.goal_dim = envs[0].unwrapped.goal_dim
        
        self.action_space_shape = envs[0].action_space.shape
        

        # Configure acmodel

        self.hi_policy_value_net.to(self.device)
        self.hi_policy_value_net.train()
        self.lo_policy_value_net.to(self.device)
        self.lo_policy_value_net.train()

        self.lo_optimizer = torch.optim.Adam(self.lo_policy_value_net.parameters(), self.lr, eps=self.optim_eps)
        self.hi_optimizer = torch.optim.Adam(self.hi_policy_value_net.parameters(), self.hi_lr, eps=self.optim_eps)

        ## Initialize logs =================================
        self.obs = self.env.reset()

        # Store helpers values
        self.num_procs = len(envs)
        self.num_frames_lo = (self.num_frames_per_proc - 1) * self.num_procs
        
        lo_shape = (self.num_frames_per_proc, self.num_procs)
        lo_act_shape = lo_shape + self.action_space_shape
        lo_goal_shape = lo_shape + (self.goal_dim,)

        self.rewards = torch.zeros(*lo_shape, device=self.device)
        self.lo_obss = [None]*(lo_shape[0])
        self.lo_goals = torch.zeros(*lo_goal_shape, device=self.device)
        self.lo_mask = torch.ones(lo_shape[1], device=self.device)
        self.lo_masks = torch.zeros(*lo_shape, device=self.device)
        self.lo_actions = torch.zeros(*lo_act_shape, device=self.device)
        self.lo_values = torch.zeros(*lo_shape, device=self.device)
        self.lo_rewards = torch.zeros(*lo_shape, device=self.device)
        self.lo_advantages = torch.zeros(*lo_shape, device=self.device)
        self.lo_log_probs = torch.zeros(*lo_act_shape, device=self.device)



        self.hi_mask = torch.zeros(self.num_procs, device=self.device)
        self.hi_reward = torch.zeros(self.num_procs, device=self.device)

        self.hi_obss = [[] for i in range(self.num_procs)]
        self.hi_actions = [[] for i in range(self.num_procs)]
        self.hi_action_masks = [[] for i in range(self.num_procs)]
        self.hi_log_probs = [[] for i in range(self.num_procs)]
        self.hi_values = [[] for i in range(self.num_procs)]
        self.hi_rewards = [[] for i in range(self.num_procs)]
        self.hi_masks = [[self.hi_mask[i].clone()] for i in range(self.num_procs)]
        self.hi_advantages = [[] for i in range(self.num_procs)]


        self.cur_goal = torch.zeros(self.num_procs, self.goal_dim, device=self.device)


        # self.hi_obss = [None]*(hi_shape[0])
        # self.hi_mask = torch.ones(hi_shape[1], device=self.device)
        # self.hi_masks = torch.zeros(*hi_shape, device=self.device)
        # self.hi_actions = torch.zeros(*hi_act_shape, device=self.device)#, dtype=torch.int)
        # self.hi_log_probs = torch.zeros(*hi_act_shape, device=self.device)
        # self.hi_values = torch.zeros(*hi_shape, device=self.device)
        # self.hi_rewards = torch.zeros(*hi_shape, device=self.device)
        # self.hi_advantages = torch.zeros(*hi_shape, device=self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_diversity = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_prediction = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_diversity = [0] * self.num_procs
        self.log_prediction = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

    ## Import policy methods
    from torch_ac.algos._hier_policy_opt import collect_experiences, update_parameters, update_lo_parameters, update_hi_parameters, \
        _get_batches_starting_indexes_lo, _get_batches_starting_indexes_hi
