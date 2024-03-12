from collections import namedtuple
from typing import Optional

import numpy as np

from rlrs.datasets.movielens import MovieLens

UserStateInfo = namedtuple(
    "UserStateInfo", ["user_id", "prev_pos_items", "done", "reward"])


class MovieLenOfflineEnv:
    """ """

    def __init__(self, db: MovieLens, state_size: int, user_id: Optional[int] = None):
        self.db = db
        self.state_size = state_size
        self.user_id = user_id
        self.avail_users = self.db.filter_user_by_history(
            lambda x: x > state_size)

        self.reset()
        self.done_count = 3000

    def reset(self) -> UserStateInfo:
        """
        Reset the env.
        Choose a specific user or randomly select from
        a pool of users.
        Recommended items : rated (>=4) movies.

        Returns:
            UserStateInfo indicates:
                 1. which user is considering (`user_id`)
                 2. historical positive items (`prev_positive_items`)
                 3. Whether we go through the whole session (`done`) .
                 4. Reward value
        """
        self.user = self.user_id or np.random.choice(self.avail_users)

        ratings = self.db.get_ratings(self.user)

        self.user_ratings = {
            r.MovieID: r.Rating for r in ratings.itertuples(index=False)
        }
        # historical positive items
        self.prev_postive_items = ratings[: self.state_size]["MovieID"]
        # assuming all previous items are recommended by the agent
        self.recommended_items = set(self.prev_postive_items)
        self.done = False
        return UserStateInfo(self.user, self.prev_postive_items, self.done, 0)

    def step(self, new_rec_item) -> UserStateInfo:
        """
        Given a new recommended item, the env calculate
        the updated reward, depending whether the agent
        correctly recommend the item or not.
        Returns:
            UserStateInfo indicates:
                 1. which user is considering (`user_id`)
                 2. historical positive items (`prev_positive_items`)
                 3. Whether we go through the whole session (`done`) .
                 4. Reward value
        """
        ...

    def get_item_names(self, item_ids):
        # should call db
        ...

    def get_positive_items(self, user_idx): ...

    @property
    def num_users(self):
        return self.db.num_users

    @property
    def num_items(self):
        return self.db.num_items

    def users(self):
        """
        Return all users in the env
        """
        ...

    def items(self):
        """
        Return all items in the env
        """
        ...
