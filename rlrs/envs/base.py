from abc import ABC, abstractmethod
from typing import List

from rlrs.datasets.base import Dataset


class OfflineEnvBase(ABC):
    def __init__(self, db: Dataset):
        self.db = db
        self.avail_users = []

    @abstractmethod
    def reset(self): ...

    @abstractmethod
    def step(self, recommended_items): ...

    @property
    def num_users(self) -> int:
        return len(self.avail_users)

    @property
    def num_items(self) -> int:
        return self.db.num_items

    @property
    def users(self) -> List:
        """
        Return all users in the env
        """
        return self.db.users[self.db.users.index.isin(self.avail_users)]

    @property
    def items(self):
        """
        Return all items (dataframe) in the env
        """
        return self.db.items

    @property
    def database(self):
        """
        Return the underlying database of the environment
        """
        return self.db
