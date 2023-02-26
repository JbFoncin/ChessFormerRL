from math import sqrt

import torch as t
from torch import nn


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
        
    def forward(self, hidden_state_query, hidden_state_key, hidden_state_value, attention_mask=None):
        """
        Args:
            hidden_state_query (torch.tensor): the hidden state to be projected as query
            hidden_state_key (torch.tensor): the hidden state to be projected as key
            hidden_state_value (torch.tensor): the hidden state to be projected as value
            attention_mask (torch.tensor): the attention mask to avoid attention on padding tokens

        Returns:
            torch.tensor: hidden state for the next layer
        """
        query = self.query_projector(hidden_state_query) 
        key = self.key_projector(hidden_state_key)
        value = self.value_projector(hidden_state_value)
        
        bs = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        query  = query.view(bs, seq_len_q, self.nb_head, self.dim_per_head).transpose(1, 2)
        key = key.view(bs, seq_len_k, self.nb_head, self.dim_per_head).transpose(1, 2)
        value = value.view(bs, seq_len_k, self.nb_head, self.dim_per_head).transpose(1, 2)
        
        attn_scores = query @ key.transpose(-1, -2) / sqrt(self.dim_per_head)
        
        attn = self.softmax(attn_scores)
        
        if attention_mask is not None:
            attention_mask_unsqueezed = attention_mask.unsqueeze(1)
            mask_all_heads = t.repeat_interleave(attention_mask_unsqueezed, self.nb_head, dim=1)
            attn.masked_fill_(mask_all_heads, 0.0)
            
        attn_product = attn @ value
        
        attn_product = attn_product.transpose(1, 2).contiguous().view(bs, seq_len_q, -1)
        
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
        
        
        