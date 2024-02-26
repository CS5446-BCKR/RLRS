"""
TODO:
- Add rating threshold
- Add users size sampling
"""

import pandas as pd
from path import Path
from typing import Callable


class MovieLens:

    def __init__(
        self, movies: pd.DataFrame, users: pd.DataFrame, ratings: pd.DataFrame
    ):

        self.movies: pd.DataFrame = movies.set_index("MovieID")
        self.users: pd.DataFrame = users
        self.ratings: pd.DataFrame = ratings

        self.id2movies = self.movies.to_dict("index")
        self.freq = self.ratings[["UserID", "MovieID"]
                                 ].groupby("UserID").agg(len)

    def id2movie(self, index: int):
        return self.id2movies[index]

    def filter_users_by_num_ratings(self, key: Callable):
        return self.freq.index[self.freq["MovieID"].apply(key)].tolist()

    @classmethod
    def from_folder(cls, src: Path):
        return cls(
            pd.read_csv(src / "movies.csv"),
            pd.read_csv(src / "users.csv"),
            pd.read_csv(src / "ratings.csv"),
        )
