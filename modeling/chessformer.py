from torch import nn

from .layers import (ActionDecoderLayer,
                     BoardEncoderLayer,
                     ChessFormerDecoderEmbedding,
                     ChessFormerEncoderEmbedding)


class ChessFormer(nn.Module):
    """
    The transformer model
    """
    def __init__(self, nb_encoder_layers, nb_decoder_layers, embedding_dim,
                 bottleneck_hidden_dim, dim_per_head, nb_head):
        """
        Args:
            nb_encoder_layers (int): number of encoder layers
            nb_decoder_layers (int): number of decoder layers
            embedding_dim (int): embedding size
            bottleneck_hidden_dim (int): hidden size in bottleneck
            dim_per_head (int): head size
            nb_head (int): number of heads
        """
        
        super().__init__()
        
        self.encoder_embeddings = ChessFormerEncoderEmbedding(embedding_dim)
        
        encoder = [BoardEncoderLayer(embedding_dim,
                                     nb_head,
                                     dim_per_head,
                                     bottleneck_hidden_dim)
                   for _ in range(nb_encoder_layers)]
        self.encoder = nn.ModuleList(encoder)
        
        self.final_encoder_ln = nn.LayerNorm(embedding_dim)        
        
        self.decoder_embeddings = ChessFormerDecoderEmbedding(embedding_dim)
        
        decoder = [ActionDecoderLayer(embedding_dim,
                                      nb_head,
                                      dim_per_head,
                                      bottleneck_hidden_dim)
                   for _ in range(nb_decoder_layers)]
        self.decoder = nn.ModuleList(decoder)
                
        self.q_scorer = nn.Linear(embedding_dim, 1)
        
    def forward(self, pieces_ids, colors_ids, start_move_indexes, end_move_indexes,
                decoder_attention_mask=None, target_mask=None):
        """
        Args:
            pieces_ids (torch.tensor[torch.Long]): id of each piece
            colors_ids (torch.tensor[torch.Long]): color of each piece (0 or 1)
            start_move_indexes (torch.tensor[torch.Long]): start move for each possible action
            end_move_indexes (torch.tensor[torch.Long]): destination for each possible action
            decoder_attention_mask (torch.tensor[torch.Bool]): padding mask to avoid attention on padded token for decoder
            target_mask (torch.tensor[torch.Bool]): mask for targets to be set at -inf 

        Returns:
            torch.tensor: a tensor of size (possible_actions)
        """
        
        hidden_state_encoder = self.encoder_embeddings(pieces_ids, colors_ids)
        
        for encoder_layer in self.encoder:
            hidden_state_encoder = encoder_layer(hidden_state_encoder)
            
        hidden_state_encoder = self.final_encoder_ln(hidden_state_encoder) # I don't know if it's really useful
        
        hidden_state_decoder = self.decoder_embeddings(start_move_indexes, end_move_indexes)
        
        for decoder_layer in self.decoder:
            hidden_state_decoder = decoder_layer(hidden_state_encoder,
                                                 hidden_state_decoder,
                                                 attention_mask=decoder_attention_mask)
            
        q_scores = self.q_scorer(hidden_state_decoder)
        
        if target_mask is not None:
            q_scores[target_mask] = float('-inf')
        
        return q_scores
    
    