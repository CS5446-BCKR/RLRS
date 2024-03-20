"""
TODO (02/03/24): save and load networks
"""

import torch
from loguru import logger
from omegaconf import DictConfig
from path import Path
from torch import nn
from torch.optim import Adam, lr_scheduler

from .utils import soft_replace_update


class ActorModel(nn.Module):
    """
    Actor Model, or the policy network creates *actions* (not real action through),
    it needs to multiply with item embeddings to get the ranked score for recommended items.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(ActorModel, self).__init__()
        self.layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        ]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, inputs):
        return self.layers(inputs)


class Actor(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        tau,
        lr,
        step_size,
    ):
        super(Actor, self).__init__()
        self.online_network = ActorModel(input_dim, hidden_dim, output_dim)
        self.target = ActorModel(input_dim, hidden_dim, output_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.tau = tau
        self.lr = lr
        self.step_size = step_size
        # hard code optimizer here
        self.optim = Adam(self.online_network.parameters(), lr=self.lr)
        self.scheduler = lr_scheduler.StepLR(self.optim, step_size=self.step_size)

    def update_target(self):
        """
        Soft update
        Ref: https://github.com/navneet-nmk/pytorch-rl/blob/8329234822dcb977931c72db691eabf5b635788c/models/DDPG.py#L176
        """
        soft_replace_update(self.target, self.online_network, self.tau)

    def initialize(self):
        # copy weights from online to target
        soft_replace_update(self.target, self.online_network, 1.0)

    def forward(self, inputs):
        return self.online_network(inputs)

    def target_forward(self, inputs):
        return self.target(inputs)

    def fit(self, inputs, state_grads):
        self.online_network.train()
        self.optim.zero_grad()
        outputs = self.online_network(inputs)
        outputs.backward(-state_grads)
        self.optim.step()
        self.scheduler.step()

    def save(self, save_path: Path):
        torch.save(
            {
                "online_network": self.online_network.state_dict(),
                "target": self.target.state_dict(),
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "lr": self.lr,
                "step_size": self.step_size,
                "tau": self.tau,
            },
            save_path,
        )

    def load(self, checkpoint_path: Path):
        checkpoint = torch.load(checkpoint_path)

        self.online_network.load_state_dict(checkpoint["online_network"])
        self.target.load_state_dict(checkpoint["target"])

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path):
        checkpoint = torch.load(checkpoint_path)
        model = cls(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            output_dim=checkpoint["output_dim"],
            tau=checkpoint["tau"],
            lr=checkpoint["lr"],
            step_size=checkpoint["step_size"],
        )
        model.load(checkpoint_path)
        return model

    @classmethod
    def from_config(cls, cfg: DictConfig):
        model = cls(
            input_dim=cfg["input_dim"],
            hidden_dim=cfg["hidden_dim"],
            output_dim=cfg["output_dim"],
            tau=cfg["tau"],
            lr=cfg["lr"],
            step_size=cfg["step_size"],
        )
        return model
