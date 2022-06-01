import torch
import torch.nn.functional as F

import utils
from torch.distributions import Categorical
from hier_policy_value_models import HighPolicyValueModel, LoPolicyValueModel
from dynamics_model import LatentDynamicsModel
from env_model import getHiEnvEncoder

class HierAgent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, obs_space, action_space, model_dir, num_cities, goal_dim, 
                 device=None, num_envs=1, h_dim=128):
        obs_space, self.preprocess_obss = utils.get_obss_preprocessor(obs_space)
        self.hi_policy_value_net = HighPolicyValueModel(obs_space, num_cities, h_dim)
        self.lo_policy_value_net = LoPolicyValueModel(obs_space, action_space, goal_dim, h_dim)

        self.device = device
        self.num_envs = num_envs

        self.hi_policy_value_net.load_state_dict(utils.get_hi_model_state(model_dir))
        self.hi_policy_value_net.to(self.device)
        self.hi_policy_value_net.eval()

        self.lo_policy_value_net.load_state_dict(utils.get_lo_model_state(model_dir))
        self.lo_policy_value_net.to(self.device)
        self.lo_policy_value_net.eval()

        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(utils.get_vocab(model_dir))


    def get_hi_action(self, obs, available_goals):
        with torch.no_grad():
            hi_dist, hi_value = self.hi_policy_value_net(self.preprocess_obss([obs], device=self.device))
            logits = hi_dist.logits[0]
            logits[~available_goals] = float('-inf')

            return Categorical(logits=logits).sample()

    def get_lo_action(self, obs, goal):
        with torch.no_grad():
            preprocessed_obs = self.preprocess_obss([obs], device=self.device)
            preprocessed_obs.goal = goal.to(self.device)
            dist, val = self.lo_policy_value_net(preprocessed_obs)
            action = dist.sample()

        return action.cpu().numpy()



