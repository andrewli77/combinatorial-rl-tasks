import torch, torch.nn.functional as F
from torch_ac.torch_utils import DictList, ParallelEnv

import numpy as np
import pdb
import copy
import random

## Samples the environment using the policy. Uses the standard RL framework where the policy observes each observation.
def collect_experiences(self):
    # Reset the experiences for training the reward/dynamics models
    self.hi_actions_flat = []
    self.hi_rewards_flat = []
    self.hi_values_flat = []
    self.hi_dones_flat = []
    self.hi_obss_flat = []

    # We keep track of which processes are currently running.
    # If a process finishes an episode in the middle of a skill, we set this to False
    # until the next skill starts. (Skills only start on iterations i that are 0 mod self.skill_len)
    self.proc_active = [True for i in range(self.num_procs)]    
    self.frame_counter = 0

    # Collect experiences:
    # We force all high-level skills to be selected on the same iteration.
    for i in range(self.num_frames_per_proc):
        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        # Select a new skill on each environment
        if i % self.skill_len == 0:
            # Reset all processes to active
            self.proc_active = [True for i in range(self.num_procs)]

            with torch.no_grad():
                
                hi_dist, hi_value = self.hi_policy_value_net(preprocessed_obs)
                sampled_goal = hi_dist.sample()

                ## Record experiences
                self.hi_goals[i//self.skill_len] = sampled_goal
                self.hi_obss[i//self.skill_len] = self.obs
                self.hi_values[i//self.skill_len] = hi_value
                self.hi_log_probs[i//self.skill_len] = hi_dist.log_prob(sampled_goal).sum(dim=-1)


        with torch.no_grad():
            preprocessed_obs.goal = sampled_goal
            lo_dist, lo_value = self.lo_policy_value_net(preprocessed_obs)

        action = lo_dist.sample()

        if (i+1) % self.skill_len == 0:
            obs, reward, done, _ = self.env.step(action.cpu().numpy())
        else:
            obs, reward, done, _ = self.env.step_no_reset(action.cpu().numpy())

        ## HRL Reward shaping=

        with torch.no_grad():
            # Hack to extract xy position of agent from observation
            obs_xys = preprocessed_obs.obs[:, 1:3]
            self.lo_dists_to_goal[i] = torch.pow(torch.pow(preprocessed_obs.goal - obs_xys, 2).sum(dim=-1), 0.5)
            # Update experiences values
            self.lo_obss[i] = self.obs
            self.lo_goals[i] = preprocessed_obs.goal
            self.obs = obs
            self.lo_masks[i] = self.lo_mask
            self.lo_mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.lo_actions[i] = action
            self.lo_values[i] = lo_value
            self.rewards[i] = torch.tensor(reward, device=self.device)
            
            self.lo_log_probs[i] = lo_dist.log_prob(action)

        # Update log values

        self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
        self.log_episode_reshaped_return += self.rewards[i]

        self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)
        self.frame_counter += sum(self.proc_active)

        for i, done_ in enumerate(done):
            if done_ and self.proc_active[i]:
                self.proc_active[i] = False
                self.log_done_counter += 1
                self.log_return.append(self.log_episode_return[i].item())
                self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                self.log_num_frames.append(self.log_episode_num_frames[i].item())
                self.log_prediction.append(self.log_episode_prediction[i].item())

        self.log_episode_return *= self.lo_mask
        self.log_episode_reshaped_return *= self.lo_mask
        self.log_episode_num_frames *= self.lo_mask
        self.log_episode_prediction *= self.lo_mask

    # Add advantage and return to experiences

    preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
    
    with torch.no_grad():
        next_hi_dist, next_hi_value = self.hi_policy_value_net(preprocessed_obs)
        next_sampled_goal = next_hi_dist.sample()

        preprocessed_obs.goal = next_sampled_goal
        _, next_lo_value = self.lo_policy_value_net(preprocessed_obs)

        obs_xys = preprocessed_obs.obs[:, 1:3]
        next_lo_dist_to_goal = torch.pow(torch.pow(preprocessed_obs.goal - obs_xys, 2).sum(dim=-1), 0.5)

    # Update high-level experiences
    for i_hi in reversed(range(self.num_frames_per_proc_hi)):
        _rewards = self.rewards[i_hi * self.skill_len : (i_hi+1) * self.skill_len, :]
        #_discounts = torch.pow(self.discount*torch.ones(self.skill_len), torch.arange(0, self.skill_len)).to(self.device)
        self.hi_rewards[i_hi] = _rewards.sum(dim=0) #torch.einsum('ij,i->j', _rewards, _discounts)
        next_mask = self.lo_masks[(i_hi+1) * self.skill_len] if i_hi < self.num_frames_per_proc_hi - 1 else self.lo_mask
        next_hi_value = self.hi_values[i_hi+1] if i_hi < self.num_frames_per_proc_hi - 1 else next_hi_value
        next_hi_advantage = self.hi_advantages[i_hi+1] if i_hi < self.num_frames_per_proc_hi - 1 else 0

        delta = self.hi_rewards[i_hi] + next_hi_value * next_mask - self.hi_values[i_hi]
        self.hi_advantages[i_hi] = delta + self.gae_lambda * next_hi_advantage * next_mask

    # The low-level rewards are based on the high-level advantage function. 
    for i in reversed(range(self.num_frames_per_proc)):
        next_mask = self.lo_masks[i+1] if i < self.num_frames_per_proc - 1 else self.lo_mask
        next_lo_value = self.lo_values[i+1] if i < self.num_frames_per_proc - 1 else next_lo_value
        next_lo_advantage = self.lo_advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

        # Compute low (distance-to-goal-based) reward
        next_dist = self.lo_dists_to_goal[i+1] if i < self.num_frames_per_proc - 1 else next_lo_dist_to_goal
        next_mask_skill = next_mask * ((i+1)%self.skill_len != 0)
        lo_reward = (self.lo_dists_to_goal[i] - next_dist) * next_mask_skill 

        delta = lo_reward + self.discount * next_lo_value * next_mask - self.lo_values[i]
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
    lo_exps.obs.goal = self.lo_goals.transpose(0,1).reshape((-1, 2))
        

    # Add high-level observations
    hi_exps = DictList()

    hi_exps.obs = [self.hi_obss[i][j] 
                        for j in range(self.num_procs)
                        for i in range(self.num_frames_per_proc_hi)]

    hi_exps.goal = self.hi_goals.transpose(0,1).reshape((-1, 2))
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
        "prediction_per_episode": self.log_prediction[-keep:],
        "num_frames_per_episode": self.log_num_frames[-keep:],
        "num_frames": self.frame_counter
    }

    self.log_done_counter = 0
    self.log_return = self.log_return[-self.num_procs:]
    self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
    self.log_prediction = self.log_prediction[-self.num_procs:]
    self.log_num_frames = self.log_num_frames[-self.num_procs:]
    return (lo_exps, hi_exps), logs


def update_parameters(self, exps):
    logs_hi = self.update_hi_parameters(exps)
    logs_lo = self.update_lo_parameters(exps)

    logs_combined = {}

    for k,v in logs_lo.items():
        logs_combined['lo_'+k] = v

    for k,v in logs_hi.items():
        logs_combined['hi_'+k] = v

    return logs_combined

def update_lo_parameters(self, exps):
    log_entropies = []
    log_values = []
    log_policy_losses = []
    log_value_losses = []
    log_grad_norms = []

    lo_exps = exps[0]

    # Initialize log values
    for _ in range(self.epochs):
        for inds in self._get_batches_starting_indexes_lo():
            # Initialize batch values

            batch_entropy = 0
            batch_value = 0
            batch_policy_loss = 0
            batch_value_loss = 0
            batch_loss = 0

            # Initialize memory
            sb = lo_exps[inds]

            # Compute loss
            dist, value = self.lo_policy_value_net(sb.obs)
            entropy = dist.entropy().mean()

            # ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
            delta_log_prob = dist.log_prob(sb.action) - sb.log_prob
            if (len(self.action_space_shape) == 1): # Not scalar actions (multivariate)
                delta_log_prob = torch.sum(delta_log_prob, dim=1)
            ratio = torch.exp(delta_log_prob)
            surr1 = ratio * sb.advantage
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
            policy_loss = -torch.min(surr1, surr2).mean()

            value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
            surr1 = (value - sb.returnn).pow(2)
            surr2 = (value_clipped - sb.returnn).pow(2)
            value_loss = torch.max(surr1, surr2).mean()

            loss = policy_loss \
                    - self.entropy_coef * entropy \
                    + self.value_loss_coef * value_loss \

            # Update batch values

            batch_entropy += entropy.item()
            batch_value += value.mean().item()
            batch_policy_loss += policy_loss.item()
            batch_value_loss += value_loss.item()
            batch_loss += loss

            # Update actor-critic

            self.lo_optimizer.zero_grad()
            batch_loss.backward()
            grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.lo_policy_value_net.parameters() if p.requires_grad and p.grad is not None) ** 0.5
            #torch.nn.utils.clip_grad_norm_([p for p in self.lo_policy_value_net.parameters() if p.requires_grad], self.max_grad_norm)
            self.lo_optimizer.step()

            # Update log values

            log_entropies.append(batch_entropy)
            log_values.append(batch_value)
            log_policy_losses.append(batch_policy_loss)
            log_value_losses.append(batch_value_loss)
            log_grad_norms.append(grad_norm)

    return {
        "entropy": np.mean(log_entropies),
        "value": np.mean(log_values),
        "policy_loss": np.mean(log_policy_losses),
        "value_loss": np.mean(log_value_losses),
        "grad_norm": np.mean(log_grad_norms)
    }

def update_hi_parameters(self, exps):
    hi_exps = exps[1]

    for _ in range(self.hi_epochs):
        # Initialize log values

        log_entropies = []
        log_values = []
        log_policy_losses = []
        log_value_losses = []
        log_grad_norms = []

        for inds in self._get_batches_starting_indexes_hi():
            # Initialize batch values

            batch_entropy = 0
            batch_value = 0
            batch_policy_loss = 0
            batch_value_loss = 0
            batch_loss = 0

            # Initialize memory
            sb = hi_exps[inds]

            # Compute loss
            dist, value = self.hi_policy_value_net(sb.obs)
            entropy = dist.entropy().sum(dim=-1).mean()

            # ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
            delta_log_prob = dist.log_prob(sb.goal).sum(dim=-1) - sb.log_prob

            ratio = torch.exp(delta_log_prob)
            surr1 = ratio * sb.advantage
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
            policy_loss = -torch.min(surr1, surr2).mean()

            value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
            surr1 = (value - sb.returnn).pow(2)
            surr2 = (value_clipped - sb.returnn).pow(2)
            value_loss = torch.max(surr1, surr2).mean()

            loss = policy_loss - self.hi_entropy_coef * entropy + self.hi_value_coef * value_loss

            # Update batch values

            batch_entropy += entropy.item()
            batch_value += value.mean().item()
            batch_policy_loss += policy_loss.item()
            batch_value_loss += value_loss.item()
            batch_loss += loss

            # Update actor-critic

            self.hi_optimizer.zero_grad()
            batch_loss.backward()
            grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.hi_policy_value_net.parameters() if p.requires_grad) ** 0.5
            #torch.nn.utils.clip_grad_norm_([p for p in self.hi_acmodel.parameters() if p.requires_grad], self.max_grad_norm)
            self.hi_optimizer.step()

            # Update log values

            log_entropies.append(batch_entropy)
            log_values.append(batch_value)
            log_policy_losses.append(batch_policy_loss)
            log_value_losses.append(batch_value_loss)
            log_grad_norms.append(grad_norm)

    # Log some values

    logs = {
        "entropy": np.mean(log_entropies),
        "value": np.mean(log_values),
        "policy_loss": np.mean(log_policy_losses),
        "value_loss": np.mean(log_value_losses),
        "grad_norm": np.mean(log_grad_norms)
    }

    return logs

def _get_batches_starting_indexes_lo(self):
    """Gives, for each batch, the indexes of the observations given to
    the model and the experiences used to compute the loss at first.

    First, the indexes are the integers from 0 to `self.num_frames` with a step of
    `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
    more diverse batches. Then, the indexes are splited into the different batches.

    Returns
    -------
    batches_starting_indexes : list of list of int
        the indexes of the experiences to be used at first for each batch
    """

    indexes = np.arange(0, self.num_frames_lo, 1)
    indexes = np.random.permutation(indexes)

    num_indexes = self.batch_size
    batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

    return batches_starting_indexes


def _get_batches_starting_indexes_hi(self):
    """Gives, for each batch, the indexes of the observations given to
    the model and the experiences used to compute the loss at first.
    First, the indexes are the integers from 0 to `self.num_frames` with a step of
    `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
    more diverse batches. Then, the indexes are splited into the different batches.
    Returns
    -------
    batches_starting_indexes : list of list of int
        the indexes of the experiences to be used at first for each batch
    """

    indexes = np.arange(0, self.num_frames_hi, 1)
    indexes = np.random.permutation(indexes)

    num_indexes = self.batch_size // self.skill_len
    batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

    return batches_starting_indexes