"""
TODO:
- Add rating threshold
- Add users size sampling
"""

from typing import Callable

import pandas as pd
from path import Path

USER_IDX_COL = "UserID"
MOVIE_IDX_COL = "MovieID"


class MovieLens:

    def __init__(
        self, movies: pd.DataFrame, users: pd.DataFrame, ratings: pd.DataFrame
    ):

        self.movies: pd.DataFrame = movies.set_index(MOVIE_IDX_COL)
        self.users: pd.DataFrame = users
        self.ratings: pd.DataFrame = ratings

        self.id2movies = self.movies.to_dict("index")
        self.freq = (
            self.ratings[[USER_IDX_COL, MOVIE_IDX_COL]].groupby(USER_IDX_COL).agg(len)
        )

    def id2movie(self, index: int):
        return self.id2movies[index]

    def filter_users_by_history(self, key: Callable):
        return self.freq.index[self.freq[MOVIE_IDX_COL].apply(key)].tolist()

    def get_ratings(self, user):
        return self.ratings[self.ratings.UserID == user]

    def get_positive_items(self, user, thres):
        raise NotImplementedError()

    @classmethod
    def from_folder(cls, src: Path):
        src = Path(src)
        return cls(
            pd.read_csv(src / "movies.csv"),
            pd.read_csv(src / "users.csv"),
            pd.read_csv(src / "ratings.csv"),
        )
