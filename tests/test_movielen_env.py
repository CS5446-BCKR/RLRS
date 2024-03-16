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
    assert env.db == db
    assert len(env.users) == 4
    assert set(env.users) == set([7, 8, 9, 10])
