import torch as t
from torch.nn.utils.rnn import pad_sequence

from chesstools.tools import PADDING_LM_ID

from modeling.tools.policy_gradient import PolicyGradientChunkedBatchGenerator

class A2CChunkedBatchGenerator(PolicyGradientChunkedBatchGenerator):
    """Creates chunked batch for A2C model gradient accumulation
    """
    @staticmethod
    def prepare_input_for_batch(episode_data, max_batch_size, device='cpu'):
        """
        creates training data for policy gradient model

        Args:
            episode_data (list[dict]): list where each element is a decision making in an unique episode
            max_batch_size (int): maximum size of each chunk for gradient accumulation
            device (str, optional): the device where tensor will be created. Defaults to 'cpu'.
        """
        nb_chunks = (len(episode_data) // max_batch_size) + 1
        
        pieces_ids = t.cat([ep_data['pieces_ids'] for ep_data in episode_data], dim=0).to(device)
        colors_ids = t.cat([ep_data['colors_ids'] for ep_data in episode_data], dim=0).to(device)

        starting_points = [ep_data['start_move_indexes'].squeeze(0) for ep_data in episode_data]
        starting_points_padded = pad_sequence(starting_points, batch_first=True, padding_value=PADDING_LM_ID).to(device)

        destinations = [ep_data['end_move_indexes'].squeeze(0) for ep_data in episode_data]
        destinations_padded = pad_sequence(destinations, batch_first=True, padding_value=PADDING_LM_ID).to(device)
        
        state_value_target = [ep_data['state_value_target'] for ep_data in episode_data]
        state_value_target = t.tensor(state_value_target, device=device)
        
        target_mask = (starting_points_padded == PADDING_LM_ID).to(device)
        
        targets_idx = t.tensor([ep_data['action_index'] for ep_data in episode_data], device=device)
        
        batch_size = targets_idx.size(0)
        
        chunked_batch_data_inputs = {'pieces_ids': pieces_ids.chunk(nb_chunks),
                                     'colors_ids': colors_ids.chunk(nb_chunks),
                                     'start_move_indexes': starting_points_padded.chunk(nb_chunks),
                                     'end_move_indexes': destinations_padded.chunk(nb_chunks),
                                     'target_mask': target_mask.chunk(nb_chunks)}

        chunked_batch_data_targets =  {'state_value_target': state_value_target.chunk(nb_chunks),
                                       'action_index': targets_idx.chunk(nb_chunks)}
        
        return (chunked_batch_data_inputs, chunked_batch_data_targets), batch_size