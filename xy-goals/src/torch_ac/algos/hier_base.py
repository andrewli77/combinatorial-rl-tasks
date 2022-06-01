from abc import ABC, abstractmethod
import torch
from envs.wrappers import WaitWrapper
from torch_ac.format import default_preprocess_obss
from torch_ac.torch_utils import DictList, ParallelEnv

import numpy as np

# PPO with a high level and low level policy. The high level policy only takes a step every `skill_len` steps.
class HierBaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, hi_acmodel, lo_acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, lo_entropy_coef, hi_entropy_coef,
                 em_coef, value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, skill_len=100, n_skills = 3):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        """

        # Store parameters

        self.env = ParallelEnv(envs)
        self.hi_acmodel = hi_acmodel
        self.lo_acmodel = lo_acmodel
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.hi_discount = discount ** skill_len
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.lo_entropy_coef = lo_entropy_coef
        self.hi_entropy_coef = hi_entropy_coef
        self.em_coef = em_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward
        self.action_space_shape = envs[0].action_space.shape
        self.skill_len = skill_len
        self.n_skills = n_skills
        # Control parameters

        assert self.recurrence == 1

        # Configure acmodel

        self.hi_acmodel.to(self.device)
        self.hi_acmodel.train()
        self.lo_acmodel.to(self.device)
        self.lo_acmodel.train()

        # Store helpers values
        self.num_frames_per_proc_hi = (self.num_frames_per_proc-1) // self.skill_len + 1

        self.num_procs = len(envs)
        self.num_frames_lo = self.num_frames_per_proc * self.num_procs
        self.num_frames_hi = self.num_frames_per_proc_hi * self.num_procs
        
        assert(self.num_frames_per_proc % self.skill_len == 0)
        assert(isinstance(self.env.envs[0], WaitWrapper))
        # Initialize experience values

        lo_shape = (self.num_frames_per_proc, self.num_procs)
        lo_act_shape = lo_shape + self.action_space_shape

        hi_shape = (self.num_frames_per_proc_hi, self.num_procs)
        hi_act_shape = hi_shape


        self.obs = self.env.reset()
        self.lo_obss = [None]*(lo_shape[0])
        self.lo_skills = torch.zeros(*(lo_shape + (self.n_skills,)), device=self.device)
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
        self.hi_values = torch.zeros(*hi_shape, device=self.device)
        self.hi_rewards = torch.zeros(*hi_shape, device=self.device)
        self.hi_advantages = torch.zeros(*hi_shape, device=self.device)
        self.hi_advantages_no_gae = torch.zeros(*hi_shape, device=self.device)
        self.hi_log_probs = torch.zeros(*hi_shape, device=self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """
        skills = torch.zeros(self.num_procs, self.n_skills)

        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction
            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            # We force all environments to run the same number of steps before termination. That way, all the high level policies
            # should execute on the same iteration. 
            if i % self.skill_len == 0:
                skills = torch.zeros(self.num_procs, self.n_skills, device=self.device)
                with torch.no_grad():
                    hi_dist, hi_value = self.hi_acmodel(preprocessed_obs)
                sampled_skills = hi_dist.sample()

                ## Record experiences
                
                self.hi_actions[i//self.skill_len] = sampled_skills
                self.hi_obss[i//self.skill_len] = self.obs
                self.hi_values[i//self.skill_len] = hi_value
                self.hi_log_probs[i//self.skill_len] = hi_dist.log_prob(sampled_skills)

                for j in range(self.num_procs):
                    skills[j][sampled_skills[j]] = 1

            with torch.no_grad():
                preprocessed_obs.skill = skills
                lo_dist, lo_value = self.lo_acmodel(preprocessed_obs)
            action = lo_dist.sample()

            if (i+1) % self.skill_len == 0:
                obs, reward, done, _ = self.env.step(action.cpu().numpy())
            else:
                obs, reward, done, _ = self.env.step_no_reset(action.cpu().numpy())
            # Update experiences values
            self.lo_obss[i] = self.obs
            self.lo_skills[i] = skills.detach().clone()
            self.obs = obs

            self.lo_masks[i] = self.lo_mask
            self.lo_mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.lo_actions[i] = action
            self.lo_values[i] = lo_value
            if self.reshape_reward is not None:
                self.lo_rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.lo_rewards[i] = torch.tensor(reward, device=self.device)
            self.lo_log_probs[i] = lo_dist.log_prob(action)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.lo_rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.lo_mask
            self.log_episode_reshaped_return *= self.lo_mask
            self.log_episode_num_frames *= self.lo_mask

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)

        with torch.no_grad():
            skills = torch.zeros(self.num_procs, self.n_skills, device=self.device)
            next_hi_dist, next_hi_value = self.hi_acmodel(preprocessed_obs)
            sampled_skills = next_hi_dist.sample()

            for j in range(self.num_procs):
                skills[j][sampled_skills[j]] = 1

            preprocessed_obs.skill = skills
            _, next_lo_value = self.lo_acmodel(preprocessed_obs)

        # This old code works for the original low-level rewards received.
        # for i in reversed(range(self.num_frames_per_proc)):
        #     next_mask = self.lo_masks[i+1] if i < self.num_frames_per_proc - 1 else self.lo_mask
        #     next_lo_value = self.lo_values[i+1] if i < self.num_frames_per_proc - 1 else next_lo_value
        #     next_lo_advantage = self.lo_advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

        #     delta = self.lo_rewards[i] + self.discount * next_lo_value * next_mask - self.lo_values[i]
        #     self.lo_advantages[i] = delta + self.discount * self.gae_lambda * next_lo_advantage * next_mask


        # Update high-level experiences
        for i_hi in reversed(range(self.num_frames_per_proc_hi)):
            _lo_rewards = self.lo_rewards[i_hi * self.skill_len : (i_hi+1) * self.skill_len, :]
            _discounts = torch.pow(self.discount*torch.ones(self.skill_len), torch.arange(0, self.skill_len)).to(self.device)
            self.hi_rewards[i_hi] = torch.einsum('ij,i->j', _lo_rewards, _discounts)
            next_mask = self.lo_masks[(i_hi+1) * self.skill_len] if i_hi < self.num_frames_per_proc_hi - 1 else self.lo_mask
            next_hi_value = self.hi_values[i_hi+1] if i_hi < self.num_frames_per_proc_hi - 1 else next_hi_value
            next_hi_advantage = self.hi_advantages[i_hi+1] if i_hi < self.num_frames_per_proc_hi - 1 else 0

            delta = self.hi_rewards[i_hi] + self.hi_discount * next_hi_value * next_mask - self.hi_values[i_hi]
            self.hi_advantages_no_gae[i_hi] = delta
            self.hi_advantages[i_hi] = delta + self.hi_discount * self.gae_lambda * next_hi_advantage * next_mask
        
        # Define auxiliary low-level rewards based on HAAR algorithm.
        # The low-level rewards are based on the high-level advantage function. 
        for i in reversed(range(self.num_frames_per_proc)):
            #low_aux_reward = self.hi_advantages_no_gae[i//self.skill_len] / self.skill_len

            next_mask = self.lo_masks[i+1] if i < self.num_frames_per_proc - 1 else self.lo_mask
            next_lo_value = self.lo_values[i+1] if i < self.num_frames_per_proc - 1 else next_lo_value
            next_lo_advantage = self.lo_advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.lo_rewards[i] + self.discount * next_lo_value * next_mask - self.lo_values[i]
            #delta = low_aux_reward + self.discount * next_lo_value * next_mask - self.lo_values[i]
            self.lo_advantages[i] = delta + self.discount * self.gae_lambda * next_lo_advantage * next_mask



        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        # Add low-level observations
        lo_exps = DictList()
        lo_exps.obs = [self.lo_obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        # for all tensors below, T x P -> P x T -> P * T
        lo_exps.action = self.lo_actions.transpose(0, 1).reshape((-1, ) + self.action_space_shape)
        lo_exps.value = self.lo_values.transpose(0, 1).reshape(-1)
        lo_exps.advantage = self.lo_advantages.transpose(0, 1).reshape(-1)
        lo_exps.returnn = lo_exps.value + lo_exps.advantage
        lo_exps.log_prob = self.lo_log_probs.transpose(0, 1).reshape((-1, ) + self.action_space_shape)

        lo_exps.obs = self.preprocess_obss(lo_exps.obs, device=self.device)
        
        # Add skills to low-level observations
        lo_skills_reshaped = self.lo_skills.transpose(0,1).reshape((-1, self.n_skills))
        lo_exps.obs.skill = lo_skills_reshaped

        # Add high-level observations
        hi_exps = DictList()
        hi_exps.obs = [self.hi_obss[i][j] 
                            for j in range(self.num_procs)
                            for i in range(self.num_frames_per_proc_hi)]
        hi_exps.action = self.hi_actions.transpose(0,1).reshape(-1)
        hi_exps.value = self.hi_values.transpose(0, 1).reshape(-1)
        hi_exps.advantage = self.hi_advantages.transpose(0, 1).reshape(-1)
        hi_exps.returnn = hi_exps.value + hi_exps.advantage
        hi_exps.log_prob = self.hi_log_probs.transpose(0, 1).reshape(-1)

        hi_exps.obs = self.preprocess_obss(hi_exps.obs, device=self.device)
        
        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames_lo
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return (lo_exps, hi_exps), logs

    @abstractmethod
    def update_parameters(self):
        pass
