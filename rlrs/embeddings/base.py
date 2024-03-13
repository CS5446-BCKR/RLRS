"""
TODO: Return Torch tensor
"""

from abc import ABC, abstractmethod

import pandas as pd
import torch


class EmbeddingBase(ABC):
    @abstractmethod
    def fit(self, data): ...

    @abstractmethod
    def __getitem__(self, index) -> torch.Tensor: ...


class DummyEmbedding(EmbeddingBase):
    """
    Generate random matrix of size NxD
    where N is the number of data point, D is the embedding dimension.
    """

    def __init__(self, dim):
        self.dim = dim
        self.embeddings = None

    def fit(self, data):
        """
        Argument:
            data: DataFrame with index
        """
        self.embs = pd.DataFrame(index=data.index)
        self.embs["emb"] = self.embs.index.map(lambda i: torch.randn(self.dim))

    def __getitem__(self, index) -> torch.Tensor:
        assert self.embs is not None
        embs = self.embs.loc[index].values
        if isinstance(embs, torch.Tensor):
            return embs
        return torch.stack(embs.tolist())

    @property
    def all(self):
        """
        Not efficient here
        """
        return torch.stack(self.embs["emb"].values.tolist())
