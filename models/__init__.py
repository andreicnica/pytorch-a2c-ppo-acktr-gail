from models.model import CNNBase, MLPBase
from models.resnetish import Resnetish

import torch
from argparse import Namespace
import gym

MODELS = {
    "CNNBase": CNNBase,
    "MLPBase": MLPBase,
    "Resnetish": Resnetish,
}


def get_model(cfg: Namespace, obs_space: dict, action_space: gym.spaces, **kwargs) -> \
        torch.nn.Module:
    assert hasattr(cfg, "name") and cfg.name in MODELS,\
        "Please provide a valid model name."
    return MODELS[cfg.name](cfg, obs_space, action_space, **kwargs)
