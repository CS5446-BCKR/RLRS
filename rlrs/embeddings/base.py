"""
TODO: Return Torch tensor
"""

from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
import pandas as pd
import torch
from path import Path


class EmbeddingBase(ABC):
    @abstractmethod
    def fit(self, data): ...

    @abstractmethod
    def __getitem__(self, index) -> torch.Tensor: ...


class AccessWithIndexMixin:
    def __init__(self):
        self.embs = None

    def __getitem__(self, index) -> torch.Tensor:
        assert self.embs is not None
        embs = self.embs.loc[index]["emb"]
        if isinstance(embs, torch.Tensor):
            return embs
        embs = embs.tolist()
        if not embs:
            return None
        return torch.stack(embs)


class PretrainedEmbedding(AccessWithIndexMixin, EmbeddingBase):
    def __init__(self, data: torch.Tensor, index):
        super(PretrainedEmbedding).__init__()
        self.embs = pd.DataFrame(index=index)
        index2iloc = {ind: i for i, ind in enumerate(index)}
        self.embs["emb"] = self.embs.index.map(lambda ind: data[index2iloc[ind]])

    def fit(self, data): ...

    @classmethod
    def from_h5(clz, file_path: Path, object_name: str):
        import h5py

        with h5py.File(file_path, "r") as f:
            data = np.asarray(f[f"{object_name}_embedding"])
            index = list(f[f"{object_name}_index"])
            data = torch.from_numpy(data)
            return clz(data, index)


class DummyEmbedding(AccessWithIndexMixin, EmbeddingBase):
    """
    Generate random matrix of size NxD
    where N is the number of data point, D is the embedding dimension.
    """

    def __init__(self, dim):
        super(DummyEmbedding).__init__()
        self.dim = dim

    def fit(self, data):
        """
        Argument:
            data: DataFrame with index
        """
        self.embs = pd.DataFrame(index=data.index)
        self.embs["emb"] = self.embs.index.map(lambda i: torch.randn(self.dim))

    @cached_property
    def all(self):
        """
        Not efficient here
        """
        return torch.stack(self.embs["emb"].values.tolist())
