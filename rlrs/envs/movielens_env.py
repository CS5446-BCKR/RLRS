from typing import Optional
from rlrs.datasets.movielens import MovieLens


class MovieLenOfflineEnv:
    def __init__(self, db: MovieLens, state_size: int, user_id: Optional[int] = None):
        self.db = db
        self.state_size = state_size
        self.user_id = user_id
        self.avail_users = self.db.filter_user_by_num_rating(
            lambda x: x > state_size)

        self.reset()

    def reset(self): ...
