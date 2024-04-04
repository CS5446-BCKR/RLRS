from .base import EmbeddingBase
import torch
import torch.nn as nn

class MovieGenreEmbedding(nn.Module):
    def __init__(self, len_movies, len_genres, embedding_size):
        super(MovieGenreEmbedding, self).__init__()
        self.m_embedding = nn.Embedding(num_embeddings=len_movies, embedding_dim=embedding_size)
        self.g_embedding = nn.Embedding(num_embeddings=len_genres, embedding_dim=embedding_size)
        self.m_g_merge = nn.Linear(embedding_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def fit(self, x):
        movie_ids, genre_ids = x[:, 0], x[:, 1]
        memb = self.m_embedding(movie_ids)
        gemb = self.g_embedding(genre_ids)
        m_g = torch.sum(memb * gemb, dim=1, keepdim=True) # Dot product
        m_g = self.m_g_merge(m_g)
        return self.sigmoid(m_g)


class MovieUserEmbedding(nn.Module):
    def __init__(self, len_users, embedding_size):
        super(MovieUserEmbedding, self).__init__()
        self.u_embedding = nn.Embedding(num_embeddings=len_users, 
                                        embedding_dim=embedding_size)
        self.m_u_merge = nn.Linear(embedding_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def fit(self, x):
        user_ids, movie_ids = x[:, 0], x[:, 1]
        uemb = self.u_embedding(user_ids)
        m_u = torch.sum(uemb * movie_ids.unsqueeze(1), dim=1, keepdim=True)  # Dot product
        m_u = self.m_u_merge(m_u)
        return self.sigmoid(m_u)
