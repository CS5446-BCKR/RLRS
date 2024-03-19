from omegaconf import OmegaConf

from rlrs.datasets.movielens import MovieLens
from rlrs.envs.offline_env import OfflineEnv
from rlrs.movie_recommender import MovieRecommender

CFG = "configs/movielen_base.yaml"


def test_init_framework():
    cfg = OmegaConf.load(CFG)
    dataset = MovieLens.from_folder(cfg["input_data"])
    env = OfflineEnv(
        dataset,
        state_size=cfg["state_size"],
        rating_threshold=cfg["rating_threshold"],
        user_id=1,
    )
    recommender = MovieRecommender(env, cfg)
