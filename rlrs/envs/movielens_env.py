from typing import Optional

import numpy as np

from rlrs.datasets.movielens import MovieLens


class MovieLenOfflineEnv:
    def __init__(self, db: MovieLens, state_size: int, user_id: Optional[int] = None):
        self.db = db
        self.state_size = state_size
        self.user_id = user_id
        self.avail_users = self.db.filter_user_by_num_rating(
            lambda x: x > state_size)

        self.reset()
        self.done_count = 3000

    def reset(self):
        """
        Reset the env.
        Recommended items : rated (>=4) movies.
        """
        self.user = self.user_id or np.random.choice(self.avail_users)

        ratings = self.db.get_ratings(self.user)

        self.user_ratings = {
            r.MovieID: r.Rating for r in ratings.itertuples(index=False)
        }
        self.ratings_in_state = ratings[: self.state_size]["MovieID"]
        self.done = None

    def get_item_names(self, item_ids):
        # should call db
        ...

    @property
    def num_users(self):
        return self.db.num_users

    @property
    def num_items(self):
        return self.db.num_items
