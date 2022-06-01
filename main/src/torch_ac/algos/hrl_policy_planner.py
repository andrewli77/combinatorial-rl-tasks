import torch
from envs.wrappers import WaitWrapper
from torch_ac.format import default_preprocess_obss
from torch_ac.torch_utils import DictList, ParallelEnv

import numpy as np

# PPO with a high level and low level policy. The high level policy only takes a step every `skill_len` steps.
class HierPolicyAlgo:
    """The base class for RL algorithms."""

    def __init__(self, envs, hi_policy_value_net, lo_policy_value_net, inverse_model, skill_logits, device, preprocess_obss, train_lo, train_hi,
            epochs, batch_size, num_frames_per_proc, lr, gae_lambda, entropy_coef, discount, value_loss_coef, diversity_coef, clip_eps,
            hi_epochs, hi_batch_size, hi_entropy_coef, hi_value_coef, hi_lr,
            inverse_epochs, inverse_batch_size, inverse_lr,
            max_grad_norm, optim_eps, skill_len, n_skills
        ):

        # Store parameters
        self.device = device
        self.env = ParallelEnv(envs)
        self.hi_policy_value_net = hi_policy_value_net
        self.lo_policy_value_net = lo_policy_value_net
        self.inverse_model = inverse_model
        self.skill_logits = skill_logits.to(self.device)
        self.skill_logits.requires_grad = True
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.train_lo = train_lo
        self.train_hi = train_hi


        # Low policy (skill) optimization hyperparams
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_frames_per_proc = num_frames_per_proc
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.discount = discount
        self.value_loss_coef = value_loss_coef
        self.diversity_coef = diversity_coef
        self.clip_eps = clip_eps

        # High policy optimization hyperparams
        self.hi_epochs = hi_epochs
        self.hi_batch_size = hi_batch_size
        self.hi_entropy_coef = hi_entropy_coef
        self.hi_value_coef = hi_value_coef
        self.hi_lr = hi_lr
        
        # Inverse model optimization hyperparams
        self.inverse_epochs = inverse_epochs
        self.inverse_batch_size = inverse_batch_size
        self.inverse_lr = inverse_lr

        # Other hyperparams
        self.max_grad_norm = max_grad_norm
        self.optim_eps = optim_eps
        self.skill_len = skill_len 
        self.n_skills = n_skills
        
        self.action_space_shape = envs[0].action_space.shape
        

        # Configure acmodel

        self.hi_policy_value_net.to(self.device)
        self.hi_policy_value_net.train()
        self.lo_policy_value_net.to(self.device)
        self.lo_policy_value_net.train()
        self.inverse_model.to(self.device)
        self.inverse_model.train()


        # Collecting full trajectories for training the reward/dynamics models
        self.noop_obs = self.env.envs[0].noop_obs()



        self.lo_optimizer = torch.optim.Adam(self.lo_policy_value_net.parameters(), self.lr, eps=self.optim_eps)
        self.hi_optimizer = torch.optim.Adam(self.hi_policy_value_net.parameters(), self.hi_lr, eps=self.optim_eps)
        self.inverse_optimizer = torch.optim.Adam(self.inverse_model.parameters(), self.inverse_lr, eps=self.optim_eps)
        self.skill_logit_optimizer = torch.optim.Adam([self.skill_logits], 1e-2, eps=self.optim_eps)

        ## Initialize logs =================================
        self.obs = self.env.reset()

        # Store helpers values
        self.num_frames_per_proc_hi = (self.num_frames_per_proc-1) // self.skill_len + 1
        self.num_procs = len(envs)
        self.num_frames_lo = self.num_frames_per_proc * self.num_procs
        self.num_frames_hi = self.num_frames_per_proc_hi * self.num_procs
        self.frame_counter = 0
        
        assert(self.num_frames_per_proc % self.skill_len == 0)
        assert(isinstance(self.env.envs[0], WaitWrapper))

        lo_shape = (self.num_frames_per_proc, self.num_procs)
        lo_act_shape = lo_shape + self.action_space_shape
        hi_shape = (self.num_frames_per_proc_hi, self.num_procs)
        hi_act_shape = hi_shape

        self.rewards = torch.zeros(*lo_shape, device=self.device)
        self.lo_obss = [None]*(lo_shape[0])
        self.lo_skills = torch.zeros(*lo_shape, device=self.device, dtype=torch.long)
        self.lo_mask = torch.ones(lo_shape[1], device=self.device)
        self.lo_masks = torch.zeros(*lo_shape, device=self.device)
        self.lo_actions = torch.zeros(*lo_act_shape, device=self.device)#, dtype=torch.int)
        self.lo_values = torch.zeros(*lo_shape, device=self.device)
        self.lo_rewards = torch.zeros(*lo_shape, device=self.device)
        self.lo_advantages = torch.zeros(*lo_shape, device=self.device)
        self.lo_log_probs = torch.zeros(*lo_act_shape, device=self.device)

        self.hi_obss = [None]*(hi_shape[0])
        self.hi_mask = torch.ones(hi_shape[1], device=self.device)
        self.hi_masks = torch.zeros(*hi_shape, device=self.device)
        self.hi_actions = torch.zeros(*hi_act_shape, device=self.device)#, dtype=torch.int)
        self.hi_log_probs = torch.zeros(*hi_act_shape, device=self.device)
        self.hi_values = torch.zeros(*hi_shape, device=self.device)
        self.hi_rewards = torch.zeros(*hi_shape, device=self.device)
        self.hi_advantages = torch.zeros(*hi_shape, device=self.device)

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
    from torch_ac.algos._hier_policy_opt import collect_experiences, update_parameters, update_lo_parameters, update_hi_parameters, update_inverse_parameters, update_skill_prior, \
        _get_batches_starting_indexes_lo, _get_batches_starting_indexes_hi, _get_batches_starting_indexes_inverse
