from torch import nn

from .sublayers import BottleNeck, ResidualMultiHeadAttention


class BoardEncoderLayer(nn.Module):
    """
    basic transformer encoder layer
    """
    def __init__(self, 
                 embedding_dim,
                 nb_head,
                 dim_per_head,
                 bottleneck_hidden_dim):
        """
        Args:
            embedding_dim (int): embedding size
            nb_head (int): number of attention heads
            dim_per_head (int): hidden size per head
            bottleneck_intermediate_dim (int): intermediate size in bottleneck
        """
        super().__init__()
        
        self.multihead_attention = ResidualMultiHeadAttention(nb_head, dim_per_head, embedding_dim)
        
        self.bottleneck = BottleNeck(embedding_dim, bottleneck_hidden_dim)
        
        self.layernorm_1 = nn.LayerNorm(embedding_dim)
        self.layernorm_2 = nn.LayerNorm(embedding_dim)

        
    def forward(self, hidden_state):
        """
        Args:
            hidden_state (t.tensor): tensor of size (seq_len, embedding_dim)

        Returns:
            t.tensor: tensor of size (seq_len, embedding_dim)
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
                 bottleneck_hidden_dim):
        """
        Args:
            embedding_dim (int): embedding size
            nb_head (int): number of attention heads
            dim_per_head (int): hidden size per head
            bottleneck_intermediate_dim (int): intermediate size in bottleneck
        """
        super().__init__()
        
        self.multihead_attention = ResidualMultiHeadAttention(nb_head, dim_per_head, embedding_dim)
        
        self.bottleneck = BottleNeck(embedding_dim, bottleneck_hidden_dim)
        
        self.layernorm_1 = nn.LayerNorm(embedding_dim)
        self.layernorm_2 = nn.LayerNorm(embedding_dim)
        
    def forward(self, enc_output, decoder_hidden_state):
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
                                                        enc_output)
        
        decoder_hidden_state = self.layernorm_2(decoder_hidden_state)
        
        decoder_hidden_state = self.bottleneck(decoder_hidden_state)
        
        return decoder_hidden_state
        
        