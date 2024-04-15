from abc import ABC, abstractmethod
from collections import namedtuple
from typing import List, Optional

from rlrs.datasets.base import Dataset

DEFAULT_DONE_COUNT = 100

UserStateInfo = namedtuple(
    "UserStateInfo", ["user_id", "prev_pos_items", "done", "reward"]
)


class OfflineEnvBase(ABC):
    def __init__(
        self, db: Dataset, avail_users: Optional[List], state_size: int, done_count: int
    ):
        self.db = db
        self.state_size = state_size
        self.avail_users = avail_users or db.get_users_by_history(state_size + 1)
        self.done_count = done_count
        self.state_size = state_size
        self.user = None

    def set_users(self, new_users):
        self.avail_users = new_users

    @abstractmethod
    def reset(self) -> UserStateInfo: ...

    @abstractmethod
    def step(
        self, recommended_items: List, positives: Optional[List] = None
    ) -> UserStateInfo: ...

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
