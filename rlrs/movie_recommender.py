"""
Modeling
"""

from omegaconf import DictConfig


class MovieRecommender:
    def __init__(self, cfg: DictConfig): ...

    def recommend(self, user_idx): ...
