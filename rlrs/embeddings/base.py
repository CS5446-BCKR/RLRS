from abc import ABC, abstractmethod


class EmbeddingBase(ABC):
    @abstractmethod
    def fit(self, data): ...
