import random

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from pytest import fixture

from rlrs.datasets.movielens import MovieLens
from rlrs.envs.offline_env import MovieLenEnv
from rlrs.movie_recommender import MovieRecommender

CFG = "configs/movielen_small_base.yaml"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


@fixture
def recommender():
    cfg = OmegaConf.load(CFG)
    dataset = MovieLens.from_folder(cfg["input_data"])
    env = MovieLenEnv(
        dataset,
        state_size=cfg["state_size"],
        rating_threshold=cfg["rating_threshold"],
        user_id=2,
        done_count=40,
    )
    return MovieRecommender(env, cfg)


def test_init_framework(recommender):
    # dummy check
    recommender is not None


@pytest.mark.slow
def test_train_framework(recommender):
    recommender.train()
