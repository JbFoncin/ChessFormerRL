from torch import nn

from .layers import (ActionDecoderLayer, BoardEncoderLayer,
                     ChessFormerDecoderEmbedding, ChessFormerEncoderEmbedding,
                     ChessFormerEncoderEmbeddingAdvantage)


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
    The same model as DQN with softmax activation at the end
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
    
    
class ChessFormerA2C(nn.Module):
    """
    Advantage actor critic. 
    Uses two decoders, one for policy and other for action value.
    """
    def __init__(self, nb_encoder_layers, nb_decoder_layers_policy, 
                 embedding_dim, bottleneck_hidden_dim, dim_per_head, nb_head):
        """
        Args:
            nb_encoder_layers (int): number of encoder layers
            nb_decoder_layers_policy (int): number of decoder layers for policy
            embedding_dim (int): embedding size
            bottleneck_hidden_dim (int): hidden size in bottleneck
            dim_per_head (int): head size
            nb_head (int): number of heads
        """
        
        super().__init__()
        
        self.encoder_embeddings = ChessFormerEncoderEmbeddingAdvantage(embedding_dim)
        
        encoder = [BoardEncoderLayer(embedding_dim,
                                     nb_head,
                                     dim_per_head,
                                     bottleneck_hidden_dim)
                   for _ in range(nb_encoder_layers)]
        
        self.encoder = nn.ModuleList(encoder)
        
        self.state_value_linear = nn.Linear(embedding_dim, 1)
        
        self.final_encoder_ln = nn.LayerNorm(embedding_dim)        
        
        self.decoder_embeddings_policy = ChessFormerDecoderEmbedding(embedding_dim)
        
        policy_decoder = [ActionDecoderLayer(embedding_dim,
                                             nb_head,
                                             dim_per_head,
                                             bottleneck_hidden_dim)
                          for _ in range(nb_decoder_layers_policy)]
        
        self.policy_decoder = nn.ModuleList(policy_decoder)
        
        self.policy_linear = nn.Linear(embedding_dim, 1)
        
        self.policy_softmax = nn.Softmax(dim=-1)
        
        
    def forward(self, pieces_ids, colors_ids, start_move_indexes, end_move_indexes,
                target_mask=None, state_value_only=False):
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
        
        state_representation = hidden_state_encoder[:, 0, :]
        
        state_value = self.state_value_linear(state_representation)
        
        state_value = state_value.squeeze(1)
        
        if state_value_only:
            return state_value
        
        hidden_state_encoder = hidden_state_encoder[:, 1:, :]
        
        hidden_state_decoder_policy = self.decoder_embeddings_policy(start_move_indexes, end_move_indexes)
        
        for policy_decoder_layer in self.policy_decoder:
            
            hidden_state_decoder_policy = policy_decoder_layer(hidden_state_encoder,
                                                               hidden_state_decoder_policy)
            
        policy_values = self.policy_linear(hidden_state_decoder_policy)

        if target_mask is not None:
            policy_values.squeeze(2).masked_fill_(target_mask, float('-inf'))
            
        policy_scores = self.policy_softmax(policy_values.squeeze(2))
        
        return state_value, policy_scores