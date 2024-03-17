import pandas as pd
import torch

from rlrs.embeddings.base import DummyEmbedding


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

    emb = embeds[[]]
