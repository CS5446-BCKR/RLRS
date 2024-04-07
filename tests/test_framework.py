import random

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from pytest import fixture

from rlrs.datasets.food import FoodSimple
from rlrs.datasets.movielens import MovieLens
from rlrs.envs.food_offline_env import FoodOrderEnv
from rlrs.envs.offline_env import MovieLenEnv
from rlrs.movie_recommender import MovieRecommender

CFG = "configs/movielen_small_base.yaml"
AYAMPP_LITE_CFG = "configs/ayampp_small_base.yaml"
AYAMPP_USER = "zs5V5O4zMPYiKxzO0e2EBy4uq403"
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


@fixture
def ayampp_lite_recommender():
    cfg = OmegaConf.load(AYAMPP_LITE_CFG)
    dataset = FoodSimple.from_folder(cfg["input_data"])
    env = FoodOrderEnv(
        dataset,
        state_size=cfg["state_size"],
        user_id=AYAMPP_USER,
        done_count=6,
    )
    return MovieRecommender(env, cfg)


def test_init_framework(recommender):
    # dummy check
    recommender is not None


@pytest.mark.slow
def test_train_framework(recommender):
    recommender.train()


@pytest.mark.slow
def test_ayampp_train_framework(ayampp_lite_recommender):
    ayampp_lite_recommender.train()
