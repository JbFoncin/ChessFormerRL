import torch as t
from torch import nn
from math import sqrt

class ChessFormerEmbedding(nn.Module):
    """basic embedding layers
    """
    
    def __init__(self, embedding_dim):
        """
        Args:
            embedding_dim (int): size of the embeddings
        """
        
        super().__init__()
        
        self.position_emb = nn.Embedding(64, embedding_dim=embedding_dim)
        self.piece_emb = nn.Embedding(6, embedding_dim=embedding_dim)
        self.color_emb = nn.Embedding(2, embedding_dim=embedding_dim)
        
    def forward(self, pieces_ids, color_ids, indexes):
        """
        Args:
            pieces_ids (t.tensor): id of each piece
            color_ids (t.tensor): color of each piece (0 or 1)
            indexes (t.tensor): piece location on the board (0 to 63)
        """
        return self.position_emb(indexes) + self.piece_emb(pieces_ids) + self.color_emb(color_ids)
    
    
class MultiHeadAttention(nn.Module):
    """simple attention head, basic implementation
    """
    def __init__(self, nb_head, dim_per_head, emb_size):
        self.nb_head = nb_head
        self.dim_per_head = dim_per_head
        self.emb_size = emb_size
        self.query_projector = nn.Linear(emb_size, dim_per_head * nb_head)
        self.key_projector = nn.Linear(emb_size, dim_per_head * nb_head)
        self.value_projector = nn.Linear(emb_size, dim_per_head * nb_head)
        self.output = nn.Linear(dim_per_head * nb_head, emb_size)
        
    def forward(self, hidden_state):
        
        query = self.query_projector(hidden_state) 
        key = self.key_projector(hidden_state)
        value = self.value_projector(hidden_state)
        
        