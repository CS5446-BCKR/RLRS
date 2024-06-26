import numpy as np
from pytest import fixture

from rlrs.datasets.movielens import MovieLens
from rlrs.envs.offline_env import MovieLenEnv

TEST_DATA = "data/test_data/ml/"
STATE_SIZE = 3
RATING_THRESHOLD = 3
USER_ID = 9
EMPTY_HIS_USER_ID = 7


def test_loading_offline_env():
    db: MovieLens = MovieLens.from_folder(TEST_DATA)
    env: MovieLenEnv = MovieLenEnv(
        db, state_size=STATE_SIZE, rating_threshold=RATING_THRESHOLD
    )

    assert env is not None
    assert env.database == db
    assert env.num_users == 4
    assert set(env.users.index) == set([7, 8, 9, 10])
    assert env.num_items == 6
    assert set(env.items.index) == set(range(10, 16))


def test_loading_specific_env():
    db: MovieLens = MovieLens.from_folder(TEST_DATA)
    env: MovieLenEnv = MovieLenEnv(
        db, state_size=STATE_SIZE, rating_threshold=RATING_THRESHOLD, user_id=USER_ID
    )

    assert env.database == db
    assert env.num_users == 1
    assert set(env.users.index) == set([9])
    assert env.num_items == 6
    assert set(env.items.index) == set(range(10, 16))


@fixture
def env():
    db: MovieLens = MovieLens.from_folder(TEST_DATA)
    env: MovieLenEnv = MovieLenEnv(
        db, state_size=STATE_SIZE, rating_threshold=RATING_THRESHOLD, user_id=USER_ID
    )
    return env


@fixture
def neg_env():
    db: MovieLens = MovieLens.from_folder(TEST_DATA)
    env: MovieLenEnv = MovieLenEnv(
        db,
        state_size=STATE_SIZE,
        rating_threshold=RATING_THRESHOLD,
        user_id=EMPTY_HIS_USER_ID,
    )
    return env


def test_change_user(env):
    env.set_users([12])
    assert env.avail_users == [12]


def test_env_reset(env):
    state = env.reset()

    assert state.user_id == USER_ID
    assert state.prev_pos_items == [15, 11, 14]
    assert env.recommended_items == set([15, 11, 14])
    assert not state.done
    assert state.reward == 0


def test_empty_historical_positive(neg_env):
    state = neg_env.reset()
    assert state.user_id == EMPTY_HIS_USER_ID
    assert state.prev_pos_items == []
    assert neg_env.recommended_items == set([])
    assert state.done
    assert state.reward == 0


def test_env_step_positive_item(env):
    env.reset()
    state = env.step([14])
    assert np.allclose(state.reward, 1.0)


def test_env_step_negative_item(env):
    env.reset()
    state = env.step([13])
    assert np.allclose(state.reward, -0.5)
