from collections import namedtuple
from typing import List, Optional

import numpy as np

from rlrs.datasets.movielens import MovieLens

from .base import OfflineEnvBase

DEFAULT_DONE_COUNT = 100
NEGATIVE_REWARD = -0.5

UserStateInfo = namedtuple(
    "UserStateInfo", ["user_id", "prev_pos_items", "done", "reward"]
)


class MovieLenEnv(OfflineEnvBase):
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
        self.done_count = done_count

    def reset(self) -> UserStateInfo:
        """
        Reset the env:
        1. Choose a specific user or randomly select from a pool of users.
        2. Specify historical positive items (via state size)
        Recommended items : rated (>=4) movies.

        Returns:
            UserStateInfo indicates:
                 1. which user is considering (`user_id`)
                 2. historical positive items (`prev_positive_items`)
                 3. Whether we go through the whole session (`done`) .
                 4. Reward value
        """
        self.user = np.random.choice(self.avail_users)
        self.positive_items = self.db.get_positive_items(
            self.user, **{"rating_threshold": self.rating_threshold}
        )

        # historical positive items
        self.prev_positive_items = self.positive_items[: self.state_size].tolist()
        # assuming all previous items are recommended by the agent
        self.recommended_items = set(self.prev_positive_items)
        self.done = len(self.prev_positive_items) == 0
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
        assert self.user is not None

        true_positives = []
        rewards = []

        """
        Different from the paper, we allow repeated recommendation
        until the customers get annoyed.
        """

        for item in new_rec_items:
            if item in self.positive_items:
                true_positives.append(item)
                rating = self.db.get_rating(self.user, item)
                rewards.append((rating - (self.rating_threshold - 1)) / 2.0)
            else:
                # false positive
                rewards.append(NEGATIVE_REWARD)
            self.recommended_items.add(item)

        self.prev_positive_items = (
            self.prev_positive_items[len(true_positives) :] + true_positives
        )

        n_rec_items = len(self.recommended_items)
        if (
            n_rec_items > self.done_count
            or n_rec_items >= self.db.get_user_history_length(self.user)
        ):
            self.done = True
        return UserStateInfo(
            self.user, self.prev_positive_items, self.done, np.mean(rewards)
        )
