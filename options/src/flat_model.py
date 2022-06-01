import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac

from env_model import getEnvModel
from policy_network import PolicyNetwork


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module, torch_ac.ACModel):
    def __init__(self, obs_space, action_space, h_dim, distributional_value = False):
        super().__init__()

        self.env_model = getEnvModel(obs_space, h_dim)
        self.embedding_size = self.env_model.size()
        self.distributional_value = distributional_value
        self.softplus = nn.Softplus(beta=0.3)
        # Define actor's model
        self.actor = PolicyNetwork(self.embedding_size, action_space, hiddens=[h_dim], activation=nn.ReLU())
        
        # Define critic's model

        if self.distributional_value: 
            self.critic = nn.Sequential(
                nn.Linear(self.embedding_size, h_dim),
                nn.ReLU()
            )
            self.critic_mu = nn.Linear(h_dim, 1)
            self.critic_sigma = nn.Linear(h_dim, 1)

        else:
            self.critic = nn.Sequential(
                nn.Linear(self.embedding_size, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, 1)
            )
        
        # Initialize parameters correctly
        self.apply(init_params)

    def forward(self, obs):
        embedding = self.env_model(obs)

        # Actor
        dist = self.actor(embedding)

        # Critic
        if self.distributional_value:
            val_embedding = self.critic(embedding)
            mu = self.critic_mu(val_embedding)
            sigma = self.softplus(self.critic_sigma(val_embedding)) + 1e-3
            value = (mu.squeeze(1), sigma.squeeze(1))
        else:
            x = self.critic(embedding)
            value = x.squeeze(1)

        return dist, value

