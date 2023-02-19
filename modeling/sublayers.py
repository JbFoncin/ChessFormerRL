from math import sqrt

import torch as t
from torch import nn


class ChessFormerEncoderEmbedding(nn.Module):
    """
    basic embedding layer
    """
    
    def __init__(self, embedding_dim):
        """
        Args:
            embedding_dim (int): size of the embeddings
        """
        
        super().__init__()
        
        self.position_emb = nn.Embedding(64, embedding_dim=embedding_dim)
        self.piece_emb = nn.Embedding(7, embedding_dim=embedding_dim)
        self.color_emb = nn.Embedding(3, embedding_dim=embedding_dim)
        self.register_buffer('indexes', t.arange(0, 64))
        
    def forward(self, pieces_ids, color_ids):
        """
        Args:
            pieces_ids (torch.tensor): id of each piece
            color_ids (torch.tensor): color of each piece (0 or 1)
            indexes (torch.tensor): piece location on the board (0 to 63)
        """
        return self.position_emb(self.indexes) + self.piece_emb(pieces_ids) + self.color_emb(color_ids)
    
    
class ChessFormerDecoderEmbedding(nn.Module):
    """
    positional embeddings for decoder
    """
    def __init__(self, embedding_dim):
        """
        Args:
            embedding_dim (int): embedding size
        """
        super().__init__()
        
        self.initial_position_embedding = nn.Embedding(64, embedding_dim=embedding_dim)
        self.destination_embedding = nn.Embedding(64, embedding_dim=embedding_dim)
    
    def forward(self, initial_position_indexes, destination_indexes):
        """
        Args:
            initial_position_indexes (torch.tensor): tensor of starting position index of all allowed movements
            destination_indexes (torch.tensor): destination index for all allowed moves

        Returns:
            torch.tensor: tensor of dim (number of allowed moves * embedding_dim)
        """      
        return self.initial_position_embedding(initial_position_indexes) + \
               self.destination_embedding(destination_indexes)
    
class ResidualMultiHeadAttention(nn.Module):
    """
    simple attention head, basic implementation. 
    WARNING: THE INPUT HAS TO BE NORMALIZED BEFORE
    """
    def __init__(self, nb_head, dim_per_head, embedding_dim):
        """
        Args:
            nb_head (int): number of attention heads
            dim_per_head (int): hidden size of each head
            embedding_dim (int): hidden size of the model
        """
        super().__init__()
        
        self.nb_head = nb_head
        self.dim_per_head = dim_per_head
        self.embedding_dim = embedding_dim
        self.query_projector = nn.Linear(embedding_dim, dim_per_head * nb_head, bias=False)
        self.key_projector = nn.Linear(embedding_dim, dim_per_head * nb_head, bias=False)
        self.value_projector = nn.Linear(embedding_dim, dim_per_head * nb_head, bias=False)
        self.output = nn.Linear(dim_per_head * nb_head, embedding_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, hidden_state_query, hidden_state_key, hidden_state_value):
        """
        Args:
            hidden_state_query (torch.tensor): the hidden state to be projected as query
            hidden_state_key (torch.tensor): the hidden state to be projected as key
            hidden_state_value (torch.tensor): the hidden state to be projected as value

        Returns:
            torch.tensor: hidden state for the next layer
        """
        query = self.query_projector(hidden_state_query) 
        key = self.key_projector(hidden_state_key)
        value = self.value_projector(hidden_state_value)
        
        seq_len_q = query.size(0)
        seq_len_k = key.size(0)
        
        query  = query.view(seq_len_q, self.nb_head, self.dim_per_head).transpose(0, 1)
        key = key.view(seq_len_k, self.nb_head, self.dim_per_head).transpose(0, 1)
        value = value.view(seq_len_k, self.nb_head, self.dim_per_head).transpose(0, 1)
        attn_scores = query @ key.transpose(-1, -2) / sqrt(self.dim_per_head)
        attn = self.softmax(attn_scores)
        attn_product = attn @ value
        attn_product = attn_product.transpose(0, 1).contiguous().view(seq_len_q, -1)
        attn_applied = self.output(attn_product)
        
        return attn_applied + hidden_state_query
    
    
class BottleNeck(nn.Module):
    """
    basic bottleneck module
    """
    def __init__(self, embedding_dim, hidden_dim):
        """
        Args:
            embedding_dim (int): hidden size of the model
            hidden_dim (int): internal dim of the bottleneck
        """
        super().__init__()
        
        self.w_1 = nn.Linear(embedding_dim, hidden_dim) 
        self.w_2 = nn.Linear(hidden_dim, embedding_dim) 
        self.activation = nn.GELU()

    def forward(self, input_):
        """
        Args:
            input_ (torch.tensor): hidden state of the model, size (seq_len, emb_dim)

        Returns:
            torch.tensor: input processed
        """
        out = self.w_2(self.activation(self.w_1(input_)))
        
        return out + input_     
        
        
        