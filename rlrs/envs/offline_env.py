from collections import namedtuple
from typing import List, Optional

import numpy as np

from rlrs.datasets.movielens import MovieLens

DEFAULT_DONE_COUNT = 3000

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
        done_count: int = DEFAULT_DONE_COUNT,
    ):
        self.db = db
        self.state_size = state_size
        self.rating_threshold = rating_threshold
        if user_id is None:
            self.avail_users = self.db.get_users_by_history(state_size)
        else:
            self.avail_users = [user_id]

        self.user = None
        self.reset()
        assert self.user is not None
        self.done_count = done_count

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
        self.user = np.random.choice(self.avail_users)

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

        true_positives = []
        rewards = []

        for item in new_rec_items:
            if item in self.positive_items.index and item not in self.recommended_items:
                true_positives.append(item)
                rewards.append((self.db.get_rating(self.user, item) - 3) / 2.0)
            else:
                # false positive
                rewards.append(-0.5)
            self.recommended_items.add(item)
        self.prev_positive_items = (
            self.prev_positive_items[-len(true_positives) :] + true_positives
        )

        n_rec_items = len(self.recommended_items)
        # if n_rec_items > self.done_count or n_rec_items >=

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
        return self.avail_users

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
