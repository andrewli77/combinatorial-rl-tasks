import torch, torch.nn.functional as F
from torch_ac.torch_utils import DictList, ParallelEnv

import numpy as np
import pdb
import copy
import random

## Samples the environment using the policy. Uses the standard RL framework where the policy observes each observation.
def collect_experiences(self):

    # Collect experiences:
    # We force all high-level skills to be selected on the same iteration.
    for i in range(self.num_frames_per_proc):
        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)

        # === Batch observations for all envs that require choosing a new skill. ===
        obs_hi = []
        obs_hi_indices = []

        for j in range(self.num_procs):
            if self.cur_skills[j] is None:
                obs_hi.append(self.obs[j])
                obs_hi_indices.append(j)

        if len(obs_hi) > 0:
            with torch.no_grad():
                obs_hi_preprocessed = self.preprocess_obss(obs_hi, device=self.device)
                dists, vals = self.hi_policy_value_net(obs_hi_preprocessed)
                sampled_skills = dists.sample()

            for _j in range(len(obs_hi_indices)):
                j = obs_hi_indices[_j]
                self.cur_skills[j] = sampled_skills[_j]

                # Log high-level experiences
                self.hi_obss[j].append(self.obs[j])
                self.hi_actions[j].append(sampled_skills[_j])
                self.hi_values[j].append(vals[_j])
                self.hi_log_probs[j].append(dists.log_prob(sampled_skills)[_j])            

        with torch.no_grad():
            preprocessed_obs.skill = torch.tensor(self.cur_skills, device=self.device)
            lo_dist, lo_value = self.lo_policy_value_net(preprocessed_obs)

        _action = lo_dist.sample()
        action = _action[:, :-1]
        termination_prob = torch.sigmoid(_action[:, -1] * 4 - 3)

        obs, reward, done, info = self.env.step(action.cpu().numpy())

        with torch.no_grad():
            # Update experiences values
            self.lo_rewards[i] = torch.tensor(reward, device=self.device)
            self.lo_obss[i] = self.obs
            self.lo_skills[i] = torch.tensor(self.cur_skills, device=self.device)
            self.obs = obs
            self.lo_masks[i] = self.lo_mask
            self.lo_mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.lo_actions[i] = _action
            self.lo_values[i] = lo_value
            self.rewards[i] = torch.tensor(reward, device=self.device)
            self.lo_log_probs[i] = lo_dist.log_prob(_action)

            self.hi_reward += self.rewards[i]

            for j in range(self.num_procs):
                if torch.rand(()).to(self.device) < termination_prob[j]: 
                    self.hi_rewards[j].append(self.hi_reward[j].clone())
                    self.hi_reward[j] = 0.

                    self.hi_mask[j] = 0 if done[j] else 1
                    self.hi_masks[j].append(self.hi_mask[j].clone())
                    self.cur_skills[j] = None
                    self.num_terminations += 1

        # Update log values

        self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
        self.log_episode_reshaped_return += self.rewards[i]

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

    with torch.no_grad():
        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        _, next_hi_val = self.hi_policy_value_net(preprocessed_obs)


    # Process high-level experiences
    for j in range(self.num_procs):
        self.hi_advantages[j] = [0. for i_hi in range(len(self.hi_rewards[j]))]
        for i_hi in reversed(range(len(self.hi_rewards[j]))):
            next_mask = self.hi_masks[j][i_hi+1]
            next_hi_value = self.hi_values[j][i_hi+1] if i_hi+1 < len(self.hi_values[j]) else next_hi_val[j]
            next_hi_advantage = self.hi_advantages[j][i_hi+1] if i_hi < len(self.hi_rewards[j]) - 1 else 0.
            delta = self.hi_rewards[j][i_hi] + next_hi_value * next_mask - self.hi_values[j][i_hi]
            self.hi_advantages[j][i_hi] = delta + self.gae_lambda * next_hi_advantage * next_mask


    # Process low-level experiences
    for i in reversed(range(self.num_frames_per_proc-1)):
        next_mask = self.lo_masks[i+1]
        next_lo_value = self.lo_values[i+1]
        next_lo_advantage = self.lo_advantages[i+1] if i < self.num_frames_per_proc - 2 else 0

        delta = self.lo_rewards[i] + self.discount * next_lo_value * next_mask - self.lo_values[i]
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
    inverse_exps = DictList()

    if self.train_lo:
        # Add low-level observations
        lo_exps.obs = [self.lo_obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc-1)]
        # for all tensors below, T x P -> P x T -> P * T
        lo_exps.action = self.lo_actions[:-1].transpose(0, 1).reshape((-1, ) + self.action_space_shape)
        lo_exps.value = self.lo_values[:-1].transpose(0, 1).reshape(-1)
        lo_exps.advantage = self.lo_advantages[:-1].transpose(0, 1).reshape(-1)
        lo_exps.returnn = lo_exps.value + lo_exps.advantage
        lo_exps.log_prob = self.lo_log_probs[:-1].transpose(0, 1).reshape((-1, ) + self.action_space_shape)
        lo_exps.obs = self.preprocess_obss(lo_exps.obs, device=self.device)
        # Add goals to low-level observations
        lo_skills_reshaped = self.lo_skills[:-1].transpose(0,1).reshape(-1)
        lo_exps.obs.skill = lo_skills_reshaped

    # Add high-level observations
    hi_exps = DictList()

    if self.train_hi:  
        hi_exps.obs = [self.hi_obss[j][i] 
                            for j in range(self.num_procs)
                            for i in range(len(self.hi_rewards[j]))]
        hi_exps.action = torch.tensor([self.hi_actions[j][i] 
                            for j in range(self.num_procs)
                            for i in range(len(self.hi_rewards[j]))], device=self.device)
        hi_exps.value = torch.tensor([self.hi_values[j][i]
                            for j in range(self.num_procs)
                            for i in range(len(self.hi_rewards[j]))], device=self.device)
        hi_exps.advantage = torch.tensor([self.hi_advantages[j][i] 
                            for j in range(self.num_procs)
                            for i in range(len(self.hi_rewards[j]))], device=self.device)
        hi_exps.returnn = hi_exps.value + hi_exps.advantage
        hi_exps.log_prob = torch.tensor([self.hi_log_probs[j][i] 
                            for j in range(self.num_procs)
                            for i in range(len(self.hi_rewards[j]))], device=self.device)
        hi_exps.obs = self.preprocess_obss(hi_exps.obs, device=self.device)    

    self.num_frames_hi = hi_exps.action.shape[0]


    # Log some values

    keep = max(self.log_done_counter, self.num_procs)
    logs = {    
        "return_per_episode": self.log_return[-keep:],
        "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
        "num_frames_per_episode": self.log_num_frames[-keep:],
        "num_frames": self.num_frames_per_proc * self.num_procs,
        "termination_rate": self.num_terminations / (self.num_frames_per_proc * self.num_procs)
    }

    self.log_done_counter = 0
    self.log_return = self.log_return[-self.num_procs:]
    self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
    self.log_num_frames = self.log_num_frames[-self.num_procs:]


    ## Reset high-level experiences 

    self.num_terminations = 0

    for j in range(self.num_procs):
        n_remove = len(self.hi_rewards[j])
        del self.hi_obss[j][:n_remove]
        del self.hi_actions[j][:n_remove]
        del self.hi_log_probs[j][:n_remove]
        del self.hi_values[j][:n_remove]
        del self.hi_rewards[j][:n_remove]
        del self.hi_masks[j][:n_remove]
        del self.hi_advantages[j][:n_remove]

    return (lo_exps, hi_exps), logs


def update_parameters(self, exps):
    logs_lo = {}
    logs_hi = {}

    if self.train_hi:
        logs_hi = self.update_hi_parameters(exps)
    if self.train_lo:
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
            entropy = dist.entropy().mean()

            # ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
            delta_log_prob = dist.log_prob(sb.action) - sb.log_prob
            # if (len(self.act_shape) == 1): # Not scalar actions (multivariate)
            #     delta_log_prob = torch.sum(delta_log_prob, dim=1)
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

    num_indexes = self.hi_batch_size
    batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

    return batches_starting_indexes


