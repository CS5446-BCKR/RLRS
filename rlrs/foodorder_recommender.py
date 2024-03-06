"""
Food Order Modelling
"""
from omegaconf import DictConfig

class FoodOrderRecommender:
    def __init__(self, cfg: DictConfig):
        ...

    def recommend(self, user_idx):
        ...
