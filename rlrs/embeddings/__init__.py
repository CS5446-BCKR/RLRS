from omegaconf import DictConfig

from .base import DummyEmbedding, PretrainedEmbedding


def embedding_factory(cfg: DictConfig):
    name = cfg["name"]
    if name == "dummy":
        dim = cfg["dim"]
        return DummyEmbedding(dim)
    elif name == "pretrained":
        file_path = cfg["file_path"]
        object_name = cfg["object_name"]
        return PretrainedEmbedding.from_h5(file_path, object_name)

    raise RuntimeError(f"Could not identify {name} embeddings.")
