"""
Critic basically is a Deep-Q network
Estimate the Q-value given a state (from env + state network) and an action (from actor)
"""

import torch
from loguru import logger
from omegaconf import DictConfig
from path import Path
from torch import nn
from torch.autograd import grad
from torch.optim import Adam, lr_scheduler

from .utils import soft_replace_update, weighted_mse_loss


class CriticNetwork(nn.Module):
    def __init__(
        self,
        input_action_dim,
        input_state_dim,
        embedding_dim,
        hidden_dim,
    ):
        super(CriticNetwork, self).__init__()
        self.input_action_dim = input_action_dim
        self.input_state_dim = input_state_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.action_fc = nn.Sequential(
            nn.Linear(input_action_dim, embedding_dim), nn.ReLU()
        )

        self.layers = nn.Sequential(
            nn.Linear(self.embedding_dim + self.input_state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, inputs):
        """
        inputs[0]: action
        inputs[1]: state
        return: Estimated Q-value
        """
        action, state = inputs
        action_emb = self.action_fc(action)
        net_inputs = torch.cat((action_emb, state), dim=-1)
        qvalue = self.layers(net_inputs)
        return qvalue


class Critic(nn.Module):
    def __init__(
        self,
        input_action_dim,
        input_state_dim,
        embedding_dim,
        hidden_dim,
        tau,
        lr,
        step_size,
    ):
        super(Critic, self).__init__()
        self.online_network = CriticNetwork(
            input_action_dim, input_state_dim, embedding_dim, hidden_dim
        )
        self.target = CriticNetwork(
            input_action_dim, input_state_dim, embedding_dim, hidden_dim
        )
        self.input_action_dim = input_action_dim
        self.input_state_dim = input_state_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.lr = lr
        self.step_size = step_size

        self.optim = Adam(self.online_network.parameters(), lr=self.lr)
        self.scheduler = lr_scheduler.StepLR(self.optim, step_size=self.step_size)

    def update_target(self):
        soft_replace_update(self.target, self.online_network, self.tau)

    def initialize(self):
        soft_replace_update(self.target, self.online_network, 1.0)

    def fit(self, inputs, y, weights):
        self.online_network.train()
        self.optim.zero_grad()
        outputs = self.online_network(inputs)
        loss = weighted_mse_loss(outputs, y, weights)
        logger.debug(f"Online Critic Loss: {loss.item()}")
        loss.backward()
        self.optim.step()
        self.scheduler.step()

    def dq_da(self, inputs):
        """
        calc gradients w.r.t actions
        """
        action, state = inputs
        action.requires_grad = True
        state.requires_grad = True
        outputs = self.online_network.forward(inputs)
        grads = grad(
            outputs, inputs, grad_outputs=torch.ones_like(outputs), allow_unused=True
        )
        return grads[0]

    def forward(self, inputs):
        return self.online_network(inputs)

    def target_forward(self, inputs):
        return self.target(inputs)

    def calcQ(self, inputs, is_target: bool = False):
        if is_target:
            return self.target_forward(inputs)
        return self.online_network(inputs)

    def save(self, save_path: Path):
        torch.save(
            {
                "online_network": self.online_network.state_dict(),
                "target": self.target.state_dict(),
                "input_action_dim": self.input_action_dim,
                "input_state_dim": self.input_state_dim,
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
                "tau": self.tau,
                "lr": self.lr,
                "step_size": self.step_size,
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
            input_action_dim=checkpoint["input_action_dim"],
            input_state_dim=checkpoint["input_state_dim"],
            embedding_dim=checkpoint["embedding_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            tau=checkpoint["tau"],
            lr=checkpoint["lr"],
            step_size=checkpoint["step_size"],
        )
        model.load(checkpoint_path)
        return model

    @classmethod
    def from_config(cls, cfg: DictConfig):
        return cls(
            input_action_dim=cfg["input_action_dim"],
            input_state_dim=cfg["input_state_dim"],
            embedding_dim=cfg["embedding_dim"],
            hidden_dim=cfg["hidden_dim"],
            tau=cfg["tau"],
            lr=cfg["lr"],
            step_size=cfg["step_size"],
        )
