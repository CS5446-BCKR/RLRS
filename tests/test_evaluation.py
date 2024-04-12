import numpy as np
from omegaconf import OmegaConf

from rlrs.datasets.food import FoodSimple
from rlrs.envs.food_offline_env import FoodOrderEnv
from rlrs.nets.actor import Actor
from rlrs.nets.critic import Critic
from rlrs.nets.state_module import DRRAve
from rlrs.recommender import Recommender

AYAMPP_LITE_CFG = "configs/ayampp_small_base_infer.yaml"
AYAMPP_USER_1 = "zs5V5O4zMPYiKxzO0e2EBy4uq403"
CRITIC_CHECKPOINT = "data/test_data/ayampp_checkpoint/critic.pth"
ACTOR_CHECKPOINT = "data/test_data/ayampp_checkpoint/actor.pth"
DRR_CHECKPOINT = "data/test_data/ayampp_checkpoint/drr.pth"


def check_weights(old, new):
    new_param_dict = new.state_dict()
    for name, param in old.state_dict().items():
        assert name in new_param_dict
        assert np.allclose(param, new_param_dict[name])


def test_loading_food_order_recommender():
    cfg = OmegaConf.load(AYAMPP_LITE_CFG)
    dataset = FoodSimple.from_folder(cfg["input_data"])
    env = FoodOrderEnv(
        dataset, state_size=cfg["state_size"], user_id=AYAMPP_USER_1, done_count=6
    )
    rec = Recommender(env, cfg)
    old_critic = Critic.from_checkpoint(CRITIC_CHECKPOINT)
    check_weights(old_critic, rec.critic)
    old_actor = Actor.from_checkpoint(ACTOR_CHECKPOINT)
    check_weights(old_actor, rec.actor)
    old_drr = DRRAve.from_checkpoint(DRR_CHECKPOINT)
    check_weights(old_drr, rec.drr_ave)
