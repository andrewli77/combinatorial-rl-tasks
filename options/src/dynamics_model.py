import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gym 

class LatentDynamicsModel(nn.Module):
    def __init__(self, latent_dim, n_skills):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_skills = n_skills

        self.net_ = nn.Sequential(
                nn.Linear(self.latent_dim + self.n_skills + 1, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU()
            )

        self.reward_head = nn.Linear(64, 1)

        self.next_latent_head = nn.Linear(64, self.latent_dim)
        self.done_head = nn.Linear(64, 1)

    def forward(self, latent_obs, action):
        action = F.one_hot(action.long(), self.n_skills + 1)
        enc = self.net_(torch.cat([latent_obs, action], dim=1)) 
        return self.reward_head(enc), self.next_latent_head(enc), self.done_head(enc)

    @torch.no_grad()
    def forward_all_skills(self, latent_obs):
        bs = latent_obs.shape[0]
        device = latent_obs.device
        all_next_latents = torch.zeros(bs, self.n_skills, self.latent_dim, device=device)
        all_next_rewards = torch.zeros(bs, self.n_skills, 1, device=device)
        all_next_dones = torch.zeros(bs, self.n_skills, 1, device=device)

        for z in range(self.n_skills):
            action = z * torch.ones(bs).to(device)
            all_next_rewards[:, z], all_next_latents[:, z, :], all_next_dones[:, z] = self.forward(latent_obs, action)

        return all_next_rewards, all_next_latents, all_next_dones

class InverseDynamicsModel(nn.Module):
    def __init__(self, latent_dim, n_skills):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_skills = n_skills

        self.net_ = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_skills)
        )

    def forward(self, latent_obs_next):
        return self.net_(latent_obs_next)

## ========== Old code: Multiple dynamics networks ========
# # The latent dynamics model uses a separate network for each trained skill.
# # This minimizes negative interference when training several skills simultaneously
# # Each network is a `SingleSkillDynamicsModel`
# class SingleSkillDynamicsModel(nn.Module):
#     def __init__(self, latent_dim):
#         super().__init__()
#         self.latent_dim = latent_dim

#         self._net = nn.Sequential(
#                         nn.Linear(self.latent_dim, 64),
#                         nn.ReLU(),
#                         nn.Linear(64, 64),
#                         nn.ReLU(),
#                         nn.Linear(64, 64),
#                         nn.ReLU()                        
#                     )

#         self.reward_head = nn.Linear(64, 1)
#         self.next_latent_head = nn.Linear(64, self.latent_dim)
#         self.done_head = nn.Linear(64, 1)

#     def forward(self, latent_obs):
#         enc = self._net(latent_obs) 
#         return self.reward_head(enc), self.next_latent_head(enc), self.done_head(enc)

# # The latent dynamics model which can make predictions under a heterogeneous batch of skills
# class LatentDynamicsModel(nn.Module):
#     def __init__(self, latent_dim, n_skills):
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.n_skills = n_skills

#         self.nets = torch.nn.ModuleList([SingleSkillDynamicsModel(latent_dim) for i in range(self.n_skills)])

#     def forward(self, latent_obs, action):
#         bs = latent_obs.shape[0]
#         device = latent_obs.device
#         rewards = torch.zeros(bs, 1, device=device)
#         next_latents = torch.zeros(bs, self.latent_dim, device=device)
#         dones = torch.zeros(bs, 1, device=device)

#         for z in range(self.n_skills):
#             indices_z = torch.where(action == z)[0]
#             if len(indices_z) == 0:
#                 continue

#             _reward, _next_latent, _done = self.nets[z](latent_obs[indices_z])
#             rewards[indices_z] = _reward
#             next_latents[indices_z] = _next_latent 
#             dones[indices_z] = _done 

#         return rewards, next_latents, dones 

#     def forward_all_next_states(self, latent_obs):
#         bs = latent_obs.shape[0]
#         device = latent_obs.device
#         all_next_latents = torch.zeros(bs, self.n_skills, self.latent_dim, device=device)

#         for z in range(self.n_skills):
#             all_next_latents[:, z, :] = self.nets[z](latent_obs)[1]

#         return all_next_latents

