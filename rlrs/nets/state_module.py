from torch import nn
import torch

class DRRAve(nn.Module):
    """
    The DRR-ave component in the paper.
    """
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, inputs):
        """
        inputs: (user embedding, positive item embeddings)
        outputs: a vector concanating from (user, interaction, history embembdding)
        outputs size: 1x3K where K is the user/item embedding dim.
        """
        user, items = inputs
        # TODO: should to some transformation here
        history = self.avg(items)
        interaction = user * history 
        return torch.cat((user, interaction, history), -1)

