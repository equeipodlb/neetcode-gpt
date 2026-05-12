import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self._key = nn.Linear(embedding_dim, attention_dim,bias=False)
        self._query = nn.Linear(embedding_dim, attention_dim,bias=False)
        self._value = nn.Linear(embedding_dim, attention_dim,bias=False)
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        K = self._key(embedded) # (B, T, attn_dim)
        Q = self._query(embedded) # (B, T, attn_dim)
        V = self._value(embedded) # (B, T, attn_dim)
        context_length = K.shape[1]
        attn_scores = (Q @ torch.transpose(K,1,2)) / (self.attention_dim ** 0.5)
        lower_triang = torch.tril(torch.ones(context_length, context_length))
        mask = lower_triang == 0
        scores = attn_scores.masked_fill(mask, float('-inf'))
        scores = nn.functional.softmax(scores,dim=2)
        return torch.round(scores @ V, decimals=4)
