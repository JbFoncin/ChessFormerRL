from torch import nn

from .layers import (ActionDecoderLayer, BoardEncoderLayer,
                     ChessFormerDecoderEmbedding, ChessFormerEncoderEmbedding)


class ChessFormerDQN(nn.Module):
    """
    The transformer model for DQN. Supports Quantile Regression through nb_quantiles arg
    """
    def __init__(self, nb_encoder_layers, nb_decoder_layers, embedding_dim,
                 bottleneck_hidden_dim, dim_per_head, nb_head, nb_quantiles=1):
        """
        Args:
            nb_encoder_layers (int): number of encoder layers
            nb_decoder_layers (int): number of decoder layers
            embedding_dim (int): embedding size
            bottleneck_hidden_dim (int): hidden size in bottleneck
            dim_per_head (int): head size
            nb_head (int): number of heads
            nb_quantiles (int): output dim for each possible move. Default to one. 
        """
        
        super().__init__()
        
        self.nb_quantiles = nb_quantiles
        
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
        
        self.q_scorer = nn.Linear(embedding_dim, nb_quantiles)
        
        
    def forward(self, pieces_ids, colors_ids, start_move_indexes, end_move_indexes,
                target_mask=None):
        """
        Args:
            pieces_ids (torch.tensor[torch.Long]): id of each piece
            colors_ids (torch.tensor[torch.Long]): color of each piece (0 or 1)
            start_move_indexes (torch.tensor[torch.Long]): start move for each possible action
            end_move_indexes (torch.tensor[torch.Long]): destination for each possible action
            target_mask (torch.tensor[torch.Bool]): mask for targets to be set at -inf 

        Returns:
            torch.tensor: a tensor of size (batch, possible_actions)
        """
        
        hidden_state_encoder = self.encoder_embeddings(pieces_ids, colors_ids)
        
        for encoder_layer in self.encoder:
            hidden_state_encoder = encoder_layer(hidden_state_encoder)
            
        hidden_state_encoder = self.final_encoder_ln(hidden_state_encoder) # I don't know if it's really useful
        
        hidden_state_decoder = self.decoder_embeddings(start_move_indexes, end_move_indexes)
        
        for decoder_layer in self.decoder:
            hidden_state_decoder = decoder_layer(hidden_state_encoder,
                                                 hidden_state_decoder)
            
        q_scores = self.q_scorer(hidden_state_decoder)
        

        if target_mask is not None:
            q_scores.squeeze(2).masked_fill_(target_mask, float('-inf'))
        
        return q_scores.squeeze(-1)
    

class ChessFormerPolicyGradient(ChessFormerDQN):
    """
    The same model with softmax activation at the end
    """
    def __init__(self, nb_encoder_layers, nb_decoder_layers, embedding_dim,
                 bottleneck_hidden_dim, dim_per_head, nb_head):
        """
        just overloaded by adding a softmax activation attribute
        """
        super().__init__(nb_encoder_layers, nb_decoder_layers, embedding_dim,
                         bottleneck_hidden_dim, dim_per_head, nb_head, nb_quantiles=1)
        
        self.softmax = nn.Softmax(dim=-1)
        
        
    def forward(self, pieces_ids, colors_ids, start_move_indexes, end_move_indexes, target_mask=None):
        """
        Args:
            pieces_ids (torch.tensor[torch.Long]): id of each piece
            colors_ids (torch.tensor[torch.Long]): color of each piece (0 or 1)
            start_move_indexes (torch.tensor[torch.Long]): start move for each possible action
            end_move_indexes (torch.tensor[torch.Long]): destination for each possible action
            target_mask (torch.tensor[torch.Bool]): mask for targets to be set at -inf 

        Returns:
            torch.tensor: a tensor of size (batch, possible_actions)
        """
        
        output = super().forward(pieces_ids, colors_ids, start_move_indexes, end_move_indexes, target_mask)
        output_softmax = self.softmax(output)
        return output_softmax    
    