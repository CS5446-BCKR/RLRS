from typing import List, Optional

import numpy as np

from rlrs.datasets.food import FoodSimple

from .base import DEFAULT_DONE_COUNT, OfflineEnvBase, UserStateInfo

"""
Food Offline Env
"""
POSITIVE_REWARD = 1.0
NEGATIVE_REWARD = -0.1


class FoodOrderEnv(OfflineEnvBase):
    def __init__(
        self,
        db: FoodSimple,
        state_size: int,
        user_id: Optional[int] = None,
        done_count: int = DEFAULT_DONE_COUNT,
    ):

        avail_users = user_id
        if user_id is not None and not isinstance(user_id, list):
            avail_users = [user_id]

        super(FoodOrderEnv, self).__init__(db, avail_users, state_size, done_count)

    def reset(self) -> UserStateInfo:
        self.user = np.random.choice(self.avail_users)
        self.positive_items = self.db.get_positive_items(self.user)

        # historical positive items
        self.prev_positive_items = self.positive_items[: self.state_size]
        # assuming all previous items are recommended by the agent
        self.recommended_items = list(set(self.prev_positive_items))
        self.done = len(self.prev_positive_items) == 0
        return UserStateInfo(self.user, self.prev_positive_items, self.done, 0)

    def step(
        self, recommended_items: List, positives: Optional[List] = None
    ) -> UserStateInfo:
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

        positives = positives or self.positive_items

        for item in recommended_items:
            self.recommended_items.append(item)
            if item in positives:
                true_positives.append(item)
                rewards.append(POSITIVE_REWARD)
            else:
                # false positive
                rewards.append(NEGATIVE_REWARD)

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
