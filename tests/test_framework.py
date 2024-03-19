from omegaconf import OmegaConf
from pytest import fixture

from rlrs.datasets.movielens import MovieLens
from rlrs.envs.offline_env import OfflineEnv
from rlrs.movie_recommender import MovieRecommender

CFG = "configs/movielen_base.yaml"


@fixture
def recommender():
    cfg = OmegaConf.load(CFG)
    dataset = MovieLens.from_folder(cfg["input_data"])
    env = OfflineEnv(
        dataset,
        state_size=cfg["state_size"],
        rating_threshold=cfg["rating_threshold"],
        user_id=1,
    )
    return MovieRecommender(env, cfg)


def test_init_framework(recommender):
    # dummy check
    recommender is not None


def test_train_framework(recommender):
    recommender.train()
