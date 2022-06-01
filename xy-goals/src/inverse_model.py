import torch
import torch.nn as nn
import gym 
import torch.nn.functional as F


class InverseModel(nn.Module):
    def __init__(self, obs_space, n_skills, h_dim):
        super().__init__()

        assert "obs" in obs_space.keys() and "zone_obs" in obs_space.keys()
        self.obs_size = obs_space["obs"][0]
        zone_size = obs_space["zone_obs"][1]

        self.zone_net = nn.Sequential(
            nn.Linear(self.obs_size + zone_size, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),            
        )

        self.combine_net = nn.Sequential(
            nn.Linear(self.obs_size + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, n_skills)
        )

    def forward(self, obs):
        bs = obs.obs.shape[0]
        n_zones = obs.zone_obs.shape[1]

        obs_repeated = obs.obs.view(bs,1,self.obs_size).expand(bs,n_zones,self.obs_size)
        zone_enc = self.zone_net(torch.cat([obs_repeated, obs.zone_obs], dim=-1)).sum(dim=1) / n_zones
        return self.combine_net(torch.cat([obs.obs, zone_enc], dim=-1))
