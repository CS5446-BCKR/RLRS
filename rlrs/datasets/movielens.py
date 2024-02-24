import pandas as pd
from path import Path


class MovieLens:

    def __init__(
        self, movies: pd.DataFrame, users: pd.DataFrame, ratings: pd.DataFrame
    ):

        self.movies = movies.set_index("MovieID")
        self.users = users
        self.ratings = ratings

        self.id2movies = self.movies.to_dict("index")

    def id2movie(self, index: int):
        return self.id2movies[index]

    @classmethod
    def from_folder(src: Path):
        return MovieLens(
            pd.read_csv(src/"movies.csv"),
            pd.read_csv(src/"users.csv"),
            pd.read_csv(src/"ratings.csv")
        )
