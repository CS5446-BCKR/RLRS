from torch import nn


class ActorModel(nn.Module):
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
    def __init__(self, input_dim, hidden_dim, output_dim, tau):
        self.online_network = ActorModel(input_dim, hidden_dim, output_dim)
        self.target = ActorModel(input_dim, hidden_dim, output_dim)
        self.tau = tau

    def update(self): ...
