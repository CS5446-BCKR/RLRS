import numpy as np
from pytest import fixture

from rlrs.datasets.food import FoodSimple

TEST_DATA = "data/test_data/ayampp_lite/"


def test_loading_from_dataframe():
    db: FoodSimple = FoodSimple.from_folder(TEST_DATA)

    assert db.users is not None
    assert db.items is not None


@fixture
def db():
    return FoodSimple.from_folder(TEST_DATA)


def test_num_items(db):
    assert db.num_items == 8


def test_num_users(db):
    assert db.num_users == 4


def test_user_history_length(db):
    assert db.get_user_history_length("zs5V5O4zMPYiKxzO0e2EBy4uq403") == 14
    assert db.get_user_history_length("xxx") == 0


def test_get_users_by_history(db):
    users = db.get_users_by_history(2)
    assert len(users) == 3
    assert set(users) == set(
        [
            "bvkoXpIE2SMiRm6CmOUFJyWZoP62",
            "vKbZ9j6WGsdBMc11HSBLLqnS14c2",
            "zs5V5O4zMPYiKxzO0e2EBy4uq403",
        ]
    )


def test_get_rating(db):
    ratings = db.get_rating("iE7KbXeQRrgdHW20acg1NUaKlM73", "uyLAIkFrugXDFupn5cH6")
    assert ratings == 1
    ratings = db.get_rating("zs5V5O4zMPYiKxzO0e2EBy4uq403", "URP9e2B1WdCZkhX5Rl9j")
    assert ratings == 5


def test_get_positive_items(db):
    items = db.get_positive_items("bvkoXpIE2SMiRm6CmOUFJyWZoP62")
    assert items == [
        "uaaCfJnKgLDbpp5BKjJs",
        "sIYfTIFkqU3XYxoRy8G1",
        "H1jWneJIwy0vU2nTrSMJ",
    ]


def test_get_rating_matrix(db):
    ratings = db.get_rating_matrix()
    assert ratings.shape == (4, 8)
    assert np.allclose(ratings.sum(axis=1), np.ones((4,)))
    for user in db.users.index:
        for pos_item in db.get_positive_items(user):
            assert ratings.loc[user, pos_item] > 0.0
