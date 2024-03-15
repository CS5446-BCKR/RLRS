from collections import Counter
from rlrs.datasets.movielens import MovieLens


TEST_DATA = "data/test_data/ml/"


def test_loading_from_dataframe():
    db: MovieLens = MovieLens.from_folder(TEST_DATA)

    assert db.movies is not None
    assert db.users is not None
    assert db.ratings is not None


def test_filter_user_by_history():
    db: MovieLens = MovieLens.from_folder(TEST_DATA)
    avail_users = db.filter_users_by_history(lambda x: x >= 3)

    assert len(avail_users) == 4
    assert set(avail_users) == set([7, 8, 9, 10])


def test_get_ratings():
    db: MovieLens = MovieLens.from_folder(TEST_DATA)
    ratings = db.get_ratings(8)
    counter = Counter(ratings.MovieID)
    assert counter[12] == 2
    assert counter[13] == 2
    assert counter[14] == 1

def test_num_items():
    ...

def test_items():
    ...

def test_get_positive_items():
    ...
