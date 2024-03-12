from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch


class EmbeddingBase(ABC):
    @abstractmethod
    def fit(self, data): ...


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
        N = data.shape[0]
        embeddings = torch.rand((N, self.dim))
        self.embs = pd.DataFrame(index=data.index)
        self.embs["emb"] = embeddings

    def __getitem__(self, index):
        assert self.embeddings is not None
        embs = self.embs.loc[index].values
        if len(embs.shape) == 0:
            return embs.item()
        return np.stack(embs.squeeze())

    @property
    def all(self):
        return self.embs['emb'].values
