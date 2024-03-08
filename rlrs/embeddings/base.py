from abc import ABC, abstractmethod

import numpy as np


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

    def fit(self, data):
        N = data.shape[0]
        return np.random.rand(N, self.dim)
