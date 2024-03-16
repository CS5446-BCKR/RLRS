from collections import Counter

from pytest import fixture

from rlrs.datasets.movielens import MovieLens

TEST_DATA = "data/test_data/ml/"


def test_loading_from_dataframe():
    db: MovieLens = MovieLens.from_folder(TEST_DATA)

    assert db.movies is not None
    assert db.users is not None
    assert db.ratings is not None


@fixture
def db():
    return MovieLens.from_folder(TEST_DATA)


def test_filter_user_by_history(db):
    avail_users = db.get_users_by_history(3)

    assert len(avail_users) == 4
    assert set(avail_users) == set([7, 8, 9, 10])


def test_get_ratings(db):
    ratings = db.get_ratings(8)
    counter = Counter(ratings.MovieID)
    assert counter[12] == 2
    assert counter[13] == 2
    assert counter[14] == 1


def test_num_items(db):
    assert db.num_items == 6


def test_items(db):
    assert db.items is not None
    assert len(db.items) == 6


def test_get_positive_items(db):
    user_id = 9
    thres = 4
    items = db.get_positive_items(user_id, thres)

    assert len(items) == 4
    assert set(items) == set([11, 14, 15])
