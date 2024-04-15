from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from omegaconf import OmegaConf
from pydantic import BaseModel

from rlrs.datasets.food import FoodSimple
from rlrs.envs.food_offline_env import FoodOrderEnv
from rlrs.recommender import Recommender

AYAMPP_LITE_CFG = "configs/ayampp_full_deploy.yaml"
AYAMPP_USER_1 = "zs5V5O4zMPYiKxzO0e2EBy4uq403"


class Feedback(BaseModel):
    items: List[str]
    positives: List[str]


class Status(BaseModel):
    status: str

origins = [
    "*",
    "http://127.0.0.1",
    "http://127.0.0.1:8080",
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cfg = OmegaConf.load(AYAMPP_LITE_CFG)
dataset = FoodSimple.from_folder(cfg["input_data"])
env = FoodOrderEnv(
    dataset, state_size=cfg["state_size"], user_id=AYAMPP_USER_1, done_count=6
)
recommender = Recommender(env, cfg)


@app.get("/reset/{user_id}")
def reset_env(user_id: str) -> Status:
    """
    Reset the environment to the new user_id
    """
    recommender.set_users([user_id])
    recommender.reset()
    return Status(status="done")


@app.get("/recommend")
def recommend() -> List[str]:
    items = recommender.recommend()
    if items is None:
        items = []
    else:
        items = list(map(str, items))
    return items


@app.put("/feedback")
def feedback(item_feedbacks: Feedback) -> float:
    reward = recommender.feedback(item_feedbacks.items, item_feedbacks.positives)
    return reward
