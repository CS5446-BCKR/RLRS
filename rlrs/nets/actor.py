from torch import nn
from torch.optim import lr_scheduler, Adam


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
            nn.Tanh(),
        ]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, inputs):
        return self.layers(inputs)


class Actor:
    def __init__(self, input_dim, hidden_dim, output_dim, tau, lr, step_size):
        self.online_network = ActorModel(input_dim, hidden_dim, output_dim)
        self.target = ActorModel(input_dim, hidden_dim, output_dim)
        self.tau = tau
        self.lr = lr
        self.step_size = step_size
        # hard code optimizer here
        self.optim = Adam(self.online_network.parameters(), lr=self.lr)
        self.schedular = lr_scheduler.StepLR(self.optim, step_size=self.step_size)

    def update_target(self): ...

    def fit_online_network(self, states, gradients):
        """
        Fit the source network via states and gradients
        """
        self.online_network.train()
        self.optim.zero_grad()
        outputs = self.online_network(states)
        outputs.backward(gradients)
        self.scheduler.step()
