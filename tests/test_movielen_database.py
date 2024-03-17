from collections import Counter

from pytest import fixture

from rlrs.datasets.movielens import MOVIE_IDX_COL, USER_IDX_COL, MovieLens

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


def test_get_rating(db):
    ratings = db.get_rating(9, 15)
    assert ratings == 4
    ratings = db.get_rating(9, 11)
    assert ratings == 3


def test_num_items(db):
    assert db.num_items == 6


def test_items(db):
    assert db.items is not None
    assert len(db.items) == 6
    assert db.items.index.name == MOVIE_IDX_COL


def test_user_history_length(db):
    len = db.get_user_history_length(9)
    assert len == 5
    len = db.get_user_history_length(10)
    assert len == 3


def test_item_col(db):
    assert db.item_col == MOVIE_IDX_COL


def test_user_col(db):
    assert db.user_col == USER_IDX_COL
