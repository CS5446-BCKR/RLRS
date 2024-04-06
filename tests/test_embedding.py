import pandas as pd
import torch

from rlrs.embeddings.base import DummyEmbedding, PretrainedEmbedding


def test_dummy_embedding():
    embeds = DummyEmbedding(5)
    df = pd.DataFrame(index=pd.RangeIndex(2, 5))
    embeds.fit(df)

    emb = embeds[3]
    assert isinstance(emb, torch.Tensor)

    emb = embeds[[2, 4]]
    assert isinstance(emb, torch.Tensor)
    assert emb.size() == (2, 5)

    all = embeds.all
    assert isinstance(all, torch.Tensor)
    assert all.size() == (3, 5)


def test_pretrained_embedding():
    PATH = "data/test_data/embed_out/embeddings.h5"
    user_embs = PretrainedEmbedding.from_h5(PATH, "user")
    all = user_embs[[5, 6, 7, 8, 9, 10]]
    assert all.shape == (6, 5)

    item_embs = PretrainedEmbedding.from_h5(PATH, "item")
    all = item_embs[[10, 11, 12, 13, 14, 15]]
    assert all.shape == (6, 5)
