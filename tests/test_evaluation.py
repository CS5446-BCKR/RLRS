import numpy as np
import torch
from omegaconf import OmegaConf
from pytest import fixture

from rlrs.datasets.food import FoodSimple
from rlrs.envs.food_offline_env import NEGATIVE_REWARD, FoodOrderEnv
from rlrs.nets.actor import Actor
from rlrs.nets.critic import Critic
from rlrs.nets.state_module import DRRAve
from rlrs.recommender import Recommender

AYAMPP_LITE_CFG = "configs/ayampp_small_base_infer.yaml"
AYAMPP_USER_1 = "zs5V5O4zMPYiKxzO0e2EBy4uq403"
AYAMPP_USER_2 = "bvkoXpIE2SMiRm6CmOUFJyWZoP62"
CRITIC_CHECKPOINT = "data/test_data/ayampp_checkpoint/critic.pth"
ACTOR_CHECKPOINT = "data/test_data/ayampp_checkpoint/actor.pth"
DRR_CHECKPOINT = "data/test_data/ayampp_checkpoint/drr.pth"


def check_weights(old, new):
    new_param_dict = new.state_dict()
    for name, param in old.state_dict().items():
        assert name in new_param_dict
        assert np.allclose(param, new_param_dict[name])


@fixture
def rec():
    cfg = OmegaConf.load(AYAMPP_LITE_CFG)
    dataset = FoodSimple.from_folder(cfg["input_data"])
    env = FoodOrderEnv(
        dataset, state_size=cfg["state_size"], user_id=AYAMPP_USER_1, done_count=6
    )
    recommender = Recommender(env, cfg)
    return recommender


def test_loading_food_order_recommender(rec):
    old_critic = Critic.from_checkpoint(CRITIC_CHECKPOINT)
    check_weights(old_critic, rec.critic)
    old_actor = Actor.from_checkpoint(ACTOR_CHECKPOINT)
    check_weights(old_actor, rec.actor)
    old_drr = DRRAve.from_checkpoint(DRR_CHECKPOINT)
    check_weights(old_drr, rec.drr_ave)


def test_recommend_one_user(rec):
    recommended_items = list(rec.recommend_items())
    assert len(recommended_items) > 0
    assert set(recommended_items[0]) <= set(rec.items.index)


def test_recommend_multiple_uses(rec):
    rec.topk = 3
    rec_items_user_1 = list(rec.recommend_items())[0]
    # set-up env for user AYAMPP_USER_2
    rec.set_users([AYAMPP_USER_2])
    rec_items_user_2 = list(rec.recommend_items())[0]
    print(f"rec 1: {rec_items_user_1}")
    print(f"rec 2: {rec_items_user_2}")
    assert set(rec_items_user_1) != set(rec_items_user_2)

    rec.set_users([AYAMPP_USER_1])
    new_rec_items_user_1 = list(rec.recommend_items())[0]
    assert set(rec_items_user_1) == set(new_rec_items_user_1)


def test_recommend_with_feedback(rec):
    rec.topk = 2
    # 1. first, reset the env
    assert rec.user_id is None
    assert rec.prev_items is None
    assert rec.done
    assert rec.user_emb is None
    rec.reset()
    assert rec.user_id == AYAMPP_USER_1
    assert len(rec.prev_items) > 0
    assert not rec.done
    assert isinstance(rec.user_emb, torch.Tensor)

    # 2. Step func
    items = rec.recommend()
    positives = [items[0]]
    reward = rec.feedback(items, positives)
    _ = rec.recommend()
    assert reward > len(items) * NEGATIVE_REWARD
