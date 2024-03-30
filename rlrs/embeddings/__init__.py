from omegaconf import DictConfig

from .base import DummyEmbedding


def embedding_factory(cfg: DictConfig):
    name = cfg["name"]
    if name == "dummy":
        dim = cfg["dim"]
        return DummyEmbedding(dim)

    raise RuntimeError(f"Could not identify {name} embeddings.")
