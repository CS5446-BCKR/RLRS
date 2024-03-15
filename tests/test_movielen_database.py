from path import Path
from rlrs.datasets.movielens import MovieLens


TEST_DATA = "data/test_data/ml/"


def test_loading_from_dataframe():
    db: MovieLens = MovieLens.from_folder(TEST_DATA)

    assert db.movies is not None
    assert db.users is not None
    assert db.ratings is not None
