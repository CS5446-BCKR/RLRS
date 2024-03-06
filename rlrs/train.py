from omegaconf import DictConfig


class RecommenderTrainer:
    def __init__(self, rec, cfg: DictConfig):
        self.rec = rec

    def train(self):
        ...
