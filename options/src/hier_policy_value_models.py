import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import torch_ac
import gym
from env_model import getLoEnvEncoder, getHiEnvEncoder, getEnvModel
from policy_network import PolicyNetwork

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class HighPolicyValueModel(nn.Module, torch_ac.ACModel):
    def __init__(self, obs_space, n_skills, h_dim):
        super().__init__()
        self.n_skills = n_skills
        self.env_model = getEnvModel(obs_space, h_dim)
        self.embedding_size = self.env_model.size()

        self.actor = PolicyNetwork(self.embedding_size, gym.spaces.Discrete(n_skills), hiddens=[h_dim], activation=nn.ReLU())
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    def forward(self, obs):
        embedding = self.env_model(obs)

        dist = self.actor(embedding)
        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value

class LoPolicyValueModel(nn.Module, torch_ac.ACModel):
    def __init__(self, obs_space, action_space, n_skills, h_dim):
        super().__init__()
        self.n_skills = n_skills
        self.action_dim = torch.prod(torch.tensor(action_space.shape))
        #obs_space['skill'] = (n_skills,)
        self.env_model = getLoEnvEncoder(obs_space, n_skills, h_dim)
        self.embedding_size = self.env_model.size()

        # Define actor's model

        self.actor = PolicyNetwork(self.embedding_size + self.n_skills, gym.spaces.Box(low=-1., high=1., shape=(self.action_dim + 1,)), hiddens=[h_dim], activation=nn.ReLU())
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size + self.n_skills, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    def forward(self, obs):
        assert(len(obs.skill.shape)==1)
        skill = F.one_hot(obs.skill, num_classes=self.n_skills)
        embedding = self.env_model(obs)
        embedding = torch.cat([embedding, skill], dim=-1)

        dist = self.actor(embedding)
        x = self.critic(embedding)
        value = x.squeeze(1)
        return dist, value
