import os
import json
import numpy as np
import re
import torch
import torch_ac
import gym

import utils


def get_obss_preprocessor(obs_space):
    # # LidarEnv-v0
    if isinstance(obs_space, gym.spaces.Box):
        obs_space = {'obs': obs_space.shape}

        def preprocess_obss(obss, device=None):
            return torch_ac.DictList({
                'obs': torch.tensor(np.array(obss), device=device, dtype=torch.float)
            })

    # Zone-envs without hierarchy
    elif isinstance(obs_space, gym.spaces.Dict) and 'zone_obs' in obs_space.spaces.keys() and 'obs' in obs_space.spaces.keys():
        obs_space = {'zone_obs': obs_space.spaces['zone_obs'].shape, 'obs': obs_space.spaces['obs'].shape}
        def preprocess_obss(obss, device=None):
            return torch_ac.DictList({
                'zone_obs': torch.tensor(np.array([obs['zone_obs'] for obs in obss]), device=device, dtype=torch.float),
                'obs': torch.tensor(np.array([obs['obs'] for obs in obss]), device=device, dtype=torch.float),
            })

    # # Zone-envs with hierarchy 
    # elif isinstance(obs_space, gym.spaces.Dict) and 'zone_obs' in obs_space.spaces.keys() and 'hi_obs' in obs_space.spaces.keys() and 'lo_obs' in obs_space.spaces.keys():
    #     obs_space = {'zone_obs': obs_space.spaces['zone_obs'].shape, 'hi_obs': obs_space.spaces['hi_obs'].shape, 'lo_obs': obs_space.spaces['lo_obs'].shape}
    #     def preprocess_obss(obss, device=None):
    #         return torch_ac.DictList({
    #             "zone_obs": torch.tensor(np.array([obs['zone_obs'] for obs in obss]), device=device, dtype=torch.float),
    #             "hi_obs": torch.tensor(np.array([obs['hi_obs'] for obs in obss]), device=device, dtype=torch.float),
    #             "lo_obs": torch.tensor(np.array([obs['lo_obs'] for obs in obss]), device=device, dtype=torch.float)
    #         })

    # Hierarchical environments
    elif isinstance(obs_space, gym.spaces.Dict) and 'hi_obs' in obs_space.spaces.keys() and 'lo_obs' in obs_space.spaces.keys():
        obs_space = {'hi_obs': obs_space.spaces['hi_obs'].shape, 'lo_obs': obs_space.spaces['lo_obs'].shape}
        def preprocess_obss(obss, device=None):
            return torch_ac.DictList({
                "hi_obs": torch.tensor(np.array([obs['hi_obs'] for obs in obss]), device=device, dtype=torch.float),
                "lo_obs": torch.tensor(np.array([obs['lo_obs'] for obs in obss]), device=device, dtype=torch.float)
            })

    # Flatten observations
    elif isinstance(obs_space, gym.spaces.Dict):
        flat_shape = sum([np.prod(v.shape) for v in obs_space.spaces.values()])
        obs_space = {'obs': (flat_shape,)}

        def flatten_list_of_arrays(arrays):
            return [arr.flatten() for arr in arrays]
        
        def preprocess_obss(obss, device=None):
            return torch_ac.DictList({
                'obs': torch.tensor(np.array([np.concatenate(flatten_list_of_arrays(list(obs.values()))) for obs in obss]), device=device, dtype=torch.float)
            })

    # # Check if it is a MiniGrid observation space
    # elif isinstance(obs_space, gym.spaces.Dict) and list(obs_space.spaces.keys()) == ["image"]:
    #     obs_space = {"image": obs_space.spaces["image"].shape, "text": 100}

    #     vocab = Vocabulary(obs_space["text"])
    #     def preprocess_obss(obss, device=None):
    #         return torch_ac.DictList({
    #             "image": preprocess_images([obs["image"] for obs in obss], device=device),
    #             "text": preprocess_texts([obs["mission"] for obs in obss], vocab, device=device)
    #         })
    #     preprocess_obss.vocab = vocab

    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, preprocess_obss


def preprocess_images(images, device=None):
    # Bug of Pytorch: very slow if not first converted to np array
    images = np.array(images)
    return torch.tensor(images, device=device, dtype=torch.float)


def preprocess_texts(texts, vocab, device=None):
    var_indexed_texts = []
    max_text_len = 0

    for text in texts:
        tokens = re.findall("([a-z]+)", text.lower())
        var_indexed_text = np.array([vocab[token] for token in tokens])
        var_indexed_texts.append(var_indexed_text)
        max_text_len = max(len(var_indexed_text), max_text_len)

    indexed_texts = np.zeros((len(texts), max_text_len))

    for i, indexed_text in enumerate(var_indexed_texts):
        indexed_texts[i, :len(indexed_text)] = indexed_text

    return torch.tensor(indexed_texts, device=device, dtype=torch.long)


class Vocabulary:
    """A mapping from tokens to ids with a capacity of `max_size` words.
    It can be saved in a `vocab.json` file."""

    def __init__(self, max_size):
        self.max_size = max_size
        self.vocab = {}

    def load_vocab(self, vocab):
        self.vocab = vocab

    def __getitem__(self, token):
        if not token in self.vocab.keys():
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]
