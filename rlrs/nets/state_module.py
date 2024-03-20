import torch
from path import Path
from torch import nn


class DRRAve(nn.Module):
    """
    The DRR-ave component in the paper.
    """

    def __init__(self, input_dim):
        super(DRRAve, self).__init__()
        self.input_dim = input_dim
        # self.conv = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.output_dim = 3 * input_dim

    def forward(self, inputs):
        """
        inputs: (user embedding, positive item embeddings)
        outputs: a vector concanating from (user, interaction, history embembdding)
        outputs size: 1x3K where K is the user/item embedding dim.
        """
        user, items = inputs
        if len(items.size()) == 2:
            items = items.unsqueeze(1)
        history = self.avg(items.permute(2, 1, 0)).squeeze()
        interaction = user * history
        return torch.cat((user, interaction, history), -1)

    def save(self, save_path: Path):
        torch.save(
            {"input_dim": self.input_dim, "state_dict": self.state_dict()}, save_path
        )

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path):
        checkpoint = torch.load(checkpoint_path)

        model = cls(checkpoint["input_dim"])
        model.load_state_dict(checkpoint["state_dict"])
        return model
