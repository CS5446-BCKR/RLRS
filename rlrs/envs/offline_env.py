from collections import namedtuple
from typing import List, Optional

import numpy as np

from rlrs.datasets.movielens import MovieLens

UserStateInfo = namedtuple(
    "UserStateInfo", ["user_id", "prev_pos_items", "done", "reward"]
)


class OfflineEnv:
    """ """

    def __init__(
        self,
        db: MovieLens,
        state_size: int,
        rating_threshold: int,
        user_id: Optional[int] = None,
    ):
        self.db = db
        self.state_size = state_size
        self.rating_threshold = rating_threshold
        self.user_id = user_id
        self.avail_users = self.db.get_users_by_history(state_size)

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

        self.positive_items = ratings[ratings.Rating >= self.rating_threshold]

        # historical positive items
        self.prev_positive_items = self.positive_items.iloc[: self.state_size]
        # assuming all previous items are recommended by the agent
        self.recommended_items = set(self.prev_positive_items)
        self.done = False
        return UserStateInfo(self.user, self.prev_positive_items, self.done, 0)

    def step(self, new_rec_items: List) -> UserStateInfo:
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
        # Use reward function from Epn 10 in the paper
        # For non-rating positive signal, may need to find
        # alternatives

        reward = -0.5
        true_positives = []
        rewards = []

        for item in new_rec_items:
            if item in self.positive_items.index and item not in self.recommended_items:
                true_positives.append(item)
                rewards.append(self.db.get_rating(self.user, item))

    @property
    def num_users(self):
        return self.avail_users

    @property
    def num_items(self):
        return self.db.num_items

    @property
    def users(self):
        """
        Return all users in the env
        """
        return self.avail_users

    def items(self):
        """
        Return all items in the env
        """
        return self.db.items

    @property
    def database(self):
        return self.db
