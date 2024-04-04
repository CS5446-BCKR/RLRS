import torch
import torch.nn as nn

from .base import EmbeddingBase


class MenuItemEmbedding(nn.Module):
    def __init__(self, len_items, len_attributes, embedding_size):
        super(MenuItemEmbedding, self).__init__()
        self.i_embedding = nn.Embedding(
            num_embeddings=len_items, embedding_dim=embedding_size
        )
        self.a_embedding = nn.Embedding(
            num_embeddings=len_attributes, embedding_dim=embedding_size
        )
        self.i_a_merge = nn.Linear(embedding_size, 1)
        self.sigmoid = nn.Sigmoid()

    def fit(self, x):
        item_ids, attribute_ids = x[:, 0], x[:, 1]
        i_emb = self.i_embedding(item_ids)
        a_emb = self.a_embedding(attribute_ids)
        m_g = torch.sum(i_emb * a_emb, dim=1, keepdim=True)  # Dot product
        m_g = self.m_g_merge(m_g)
        return self.sigmoid(m_g)


class UserMenuItemEmbedding(nn.Module):
    def __init__(self, len_users, embedding_size):
        super(UserMenuItemEmbedding, self).__init__()
        self.u_embedding = nn.Embedding(
            num_embeddings=len_users, embedding_dim=embedding_size
        )
        self.m_u_merge = nn.Linear(embedding_size, 1)
        self.sigmoid = nn.Sigmoid()

    def fit(self, x):
        user_ids, item_ids = x[:, 0], x[:, 1]
        uemb = self.u_embedding(user_ids)
        m_u = torch.sum(
            uemb * item_ids.unsqueeze(1), dim=1, keepdim=True
        )  # Dot product
        m_u = self.m_u_merge(m_u)
        return self.sigmoid(m_u)
