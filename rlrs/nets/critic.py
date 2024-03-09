"""
Critic basically is a Deep-Q network
Estimate the Q-value given a state (from env + state network) and an action (from actor)
"""

import torch
from torch import nn
from torch.autograd import grad
from torch.optim import Adam, lr_scheduler
from utils import soft_replace_update


class CriticNetwork(nn.Module):
    def __init__(
        self,
        input_action_dim,
        input_state_dim,
        embedding_dim,
        hidden_dim,
        output_dim,
        tau,
    ):
        self.input_action_dim = input_action_dim
        self.input_state_dim = input_state_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.tau = tau

        self.action_fc = nn.Sequential(
            [nn.Linear(input_action_dim, embedding_dim), nn.ReLU()]
        )

        self.layers = nn.Sequential(
            [
                nn.Linear(self.embedding_dim +
                          self.input_state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
            ]
        )

    def forward(self, inputs):
        """
        inputs[0]: action
        inputs[1]: state
        return: Estimated Q-value
        """
        action, state = inputs
        action_emb = self.action_fc(action)
        net_inputs = torch.cat((action_emb, state), dim=1)
        qvalue = self.layers(net_inputs)
        return qvalue


class Critic:
    def __init__(
        self,
        input_action_dim,
        input_state_dim,
        embedding_dim,
        hidden_dim,
        tau,
        step_size,
        lr,
    ):
        self.online_network = Critic(
            input_action_dim, input_state_dim, embedding_dim, hidden_dim
        )
        self.target = Critic(
            input_action_dim, input_state_dim, embedding_dim, hidden_dim
        )
        self.tau = tau
        self.lr = lr
        self.step_size = step_size

        self.optim = Adam(self.online_network.parameters(), lr=self.lr)
        self.scheduler = lr_scheduler.StepLR(
            self.optim, step_size=self.step_size)

    def update_target(self):
        soft_replace_update(self.target, self.online_network, self.tau)

    def initialize(self):
        ...

    def fit_online_network(self, actions, states):
        self.online_network.train()
        self.optim.zero_grad()

    def dq_da(self, inputs):
        """
        calc gradients w.r.t actions
        """
        action, _ = inputs
        outputs = self.online_network(inputs)
        grad_action = grad(
            outputs, action, grad_outputs=torch.ones_like(outputs))
        return grad_action
