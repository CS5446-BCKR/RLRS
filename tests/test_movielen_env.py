from pytest import fixture

from rlrs.datasets.movielens import MovieLens
from rlrs.envs.offline_env import OfflineEnv

TEST_DATA = "data/test_data/ml/"
STATE_SIZE = 3
RATING_THRESHOLD = 3


def test_loading_offline_env():
    db: MovieLens = MovieLens.from_folder(TEST_DATA)
    env: OfflineEnv = OfflineEnv(
        db, state_size=STATE_SIZE, rating_threshold=RATING_THRESHOLD
    )

    assert env is not None
    assert env.database == db
    assert env.num_users == 4
    assert set(env.users) == set([7, 8, 9, 10])
    assert env.num_items == 6
    assert set(env.items.index) == set(range(10, 16))


def test_loading_specific_env():
    db: MovieLens = MovieLens.from_folder(TEST_DATA)
    env: OfflineEnv = OfflineEnv(
        db, state_size=STATE_SIZE, rating_threshold=RATING_THRESHOLD, user_id=9
    )

    assert env.database == db
    assert env.num_users == 1
    assert set(env.users) == set([9])
    assert env.num_items == 6
    assert set(env.items.index) == set(range(10, 16))


@fixture
def env():
    db: MovieLens = MovieLens.from_folder(TEST_DATA)
    env: OfflineEnv = OfflineEnv(
        db, state_size=STATE_SIZE, rating_threshold=RATING_THRESHOLD, user_id=9
    )
    return env


def test_env_reset(env): ...


def test_env_step(env): ...
