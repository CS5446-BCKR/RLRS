import numpy as np

from rlrs.embeddings.mf import SVD


def test_SVD():
    DIM = 5
    NUM_USERS = 10
    NUM_ITEMS = 12
    feedbacks = (np.random.random((NUM_USERS, NUM_ITEMS)) > 0.6).astype(np.float32)
    algo = SVD(DIM)
    algo.fit(feedbacks)
    user_embeds = algo.user_matrix
    item_embeds = algo.item_matrix

    assert user_embeds.shape == (NUM_USERS, DIM)
    assert item_embeds.shape == (NUM_ITEMS, DIM)
