import torch as t
from torch import nn

from chesstools.tools import BOARD_INDEXES

from .sublayers import BottleNeck, ResidualMultiHeadAttention


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
        self.register_buffer('indexes', t.tensor(BOARD_INDEXES, dtype=t.long))
        
    def forward(self, pieces_ids, color_ids):
        """
        Args:
            pieces_ids (torch.tensor): id of each piece
            color_ids (torch.tensor): color of each piece (0 for empty box, 1 for player piece, 2 for competitor piece)
        Returns:
            torch.tensor: tensor of dim (64 * embedding_dim)
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
        
        self.initial_position_embedding = nn.Embedding(65, embedding_dim=embedding_dim)
        self.destination_embedding = nn.Embedding(65, embedding_dim=embedding_dim)
    
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
               
               
class BoardEncoderLayer(nn.Module):
    """
    basic transformer encoder layer
    """
    def __init__(self, 
                 embedding_dim,
                 nb_head,
                 dim_per_head,
                 bottleneck_hidden_dim,
                 dropout=0.1):
        """
        Args:
            embedding_dim (int): embedding size
            nb_head (int): number of attention heads
            dim_per_head (int): hidden size per head
            bottleneck_intermediate_dim (int): intermediate size in bottleneck
        """
        super().__init__()
        
        self.multihead_attention = ResidualMultiHeadAttention(nb_head, dim_per_head, embedding_dim, dropout)
        
        self.bottleneck = BottleNeck(embedding_dim, bottleneck_hidden_dim, dropout)
        
        self.layernorm_1 = nn.LayerNorm(embedding_dim)
        self.layernorm_2 = nn.LayerNorm(embedding_dim)

        
    def forward(self, hidden_state):
        """
        Args:
            hidden_state (torch.tensor): tensor of size (seq_len, embedding_dim)

        Returns:
            torch.tensor: tensor of size (seq_len, embedding_dim)
        """        
        hidden_state = self.layernorm_1(hidden_state)
        
        hidden_state = self.multihead_attention(hidden_state, hidden_state, hidden_state)
        
        hidden_state = self.layernorm_2(hidden_state)
        
        hidden_state = self.bottleneck(hidden_state)
        
        return hidden_state
    

class ActionDecoderLayer(nn.Module):
    """
    transformer decoder to score all possible actions
    """
    def __init__(self ,
                 embedding_dim,
                 nb_head,
                 dim_per_head,
                 bottleneck_hidden_dim,
                 dropout=0.1):
        """
        Args:
            embedding_dim (int): embedding size
            nb_head (int): number of attention heads
            dim_per_head (int): hidden size per head
            bottleneck_intermediate_dim (int): intermediate size in bottleneck
        """
        super().__init__()
        
        self.multihead_attention = ResidualMultiHeadAttention(nb_head, dim_per_head, embedding_dim, dropout)
        
        self.bottleneck = BottleNeck(embedding_dim, bottleneck_hidden_dim, dropout)
        
        self.layernorm_1 = nn.LayerNorm(embedding_dim)
        self.layernorm_2 = nn.LayerNorm(embedding_dim)
        
    def forward(self, enc_output, decoder_hidden_state, attention_mask=None):
        """
        Args:
            enc_output (torch.tensor): tensor of size (number of pieces * embedding_dim)
            decoder_hidden_state (torch.tensor): tensor of size (number of possible moves * embedding dim)
        Returns:
            torch.tensor: tensor of size (q_seq_len, embedding_dim)
        """        
        decoder_hidden_state = self.layernorm_1(decoder_hidden_state)
        
        decoder_hidden_state = self.multihead_attention(decoder_hidden_state,
                                                        enc_output,
                                                        enc_output,
                                                        attention_mask)
        
        decoder_hidden_state = self.layernorm_2(decoder_hidden_state)
        
        decoder_hidden_state = self.bottleneck(decoder_hidden_state)
        
        return decoder_hidden_state
        
        