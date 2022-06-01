import torch
import torch.nn as nn
import gym 
import torch.nn.functional as F

def getHiEnvEncoder(obs_space, h_dim):
    if "zone_obs" in obs_space:
        return ZoneEnvModel(obs_space, h_dim)
    else:
        raise NotImplementedError()

def getLoEnvEncoder(obs_space, h_dim):
    return ZoneEnvGoalModel(obs_space, h_dim)

def getEnvModel(obs_space, h_dim):
    if "zone_obs" in obs_space:
        return ZoneEnvModel(obs_space, h_dim)
    else:
        raise NotImplementedError()

"""
This class is in charge of embedding the environment part of the observations.
Every environment has its own set of observations ('image', 'direction', etc) which is handeled
here by associated EnvModel subclass.

How to subclass this:
    1. Call the super().__init__() from your init
    2. In your __init__ after building the compute graph set the self.embedding_size appropriately
    3. In your forward() method call the super().forward as the default case.
    4. Add the if statement in the getEnvModel() method
"""
class EnvModel(nn.Module):
    def __init__(self, obs_space, h_dim=32):
        super().__init__()
        self.embedding_size = h_dim

    def forward(self, obs):
        return None

    def size(self):
        return self.embedding_size

class ZoneEnvModel(EnvModel):
    def __init__(self, obs_space, h_dim):
        super().__init__(obs_space)

        assert "obs" in obs_space.keys() and "zone_obs" in obs_space.keys()
        self.embedding_size = h_dim
        self.obs_size = obs_space["obs"][0]
        zone_size = obs_space["zone_obs"][1]
        

        self.zone_net_ = nn.Sequential(
            nn.Linear(self.obs_size + zone_size, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),            
        )

        self.combine_net_ = nn.Linear(self.obs_size + h_dim, h_dim)

    def forward(self, obs):
        assert "obs" in obs.keys() and "zone_obs" in obs.keys()
        bs = obs.obs.shape[0]
        n_zones = obs.zone_obs.shape[1]

        ## Encoding the constant sized obs 
        obs_repeated = obs.obs.view(bs,1,self.obs_size).expand(bs,n_zones,self.obs_size)

        ## First zone encoding layer
        zone_emb = self.zone_net_(torch.cat([obs_repeated, obs.zone_obs], dim=-1)).sum(dim=1) / n_zones
        ## Feedforward layer
        return self.combine_net_(torch.cat([obs.obs, zone_emb], dim=-1))

class ZoneEnvGoalModel(EnvModel):
    def __init__(self, obs_space, h_dim):
        super().__init__(obs_space)

        assert "obs" in obs_space.keys() and "zone_obs" in obs_space.keys() and "goal" in obs_space.keys()

        self.embedding_size = h_dim
        self.goal_dim = 2
        self.obs_size = obs_space["obs"][0]
        zone_size = obs_space["zone_obs"][1]
        

        self.zone_net_ = nn.Sequential(
            nn.Linear(self.obs_size + self.goal_dim + zone_size, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),            
        )

        self.combine_net_ = nn.Linear(self.obs_size + self.goal_dim + h_dim, h_dim)
        
    def forward(self, obs):
        assert "obs" in obs.keys() and "zone_obs" in obs.keys() and "goal" in obs.keys()

        bs = obs.obs.shape[0]
        n_zones = obs.zone_obs.shape[1]
        obs_and_goal = torch.cat([obs.obs, obs.goal], dim=-1)
        obs_repeated = obs_and_goal.view(bs,1,self.obs_size + self.goal_dim).expand(bs,n_zones,self.obs_size + self.goal_dim)

        ## First zone encoding layer
        zone_emb = self.zone_net_(torch.cat([obs_repeated, obs.zone_obs], dim=-1)).sum(dim=1) / n_zones

        ## Feedforward layer
        return self.combine_net_(torch.cat([obs_and_goal, zone_emb], dim=-1))
