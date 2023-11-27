import torch as t
from torch.nn.utils.rnn import pad_sequence

from chesstools.tools import PADDING_LM_ID


class PolicyGradientChunkedBatchGenerator:
    """
    used to iterate over chunks of batch for gradient accumulation
    """
    def __init__(self, episode_data, max_batch_size, device='cpu'):
        """
        Args:
            episode_data (list[dict]): list where each element is a decision making in an unique episode
            max_batch_size (int): maximum size of each chunk for gradient accumulation
            device (str, optional): the device where tensor will be created. Defaults to 'cpu'.
        """
        chunked_batch_data, self.batch_size = self.prepare_input_for_batch(episode_data,
                                                                           max_batch_size,
                                                                           device)
        
        self.chunked_batch_data_inputs, self.chunked_batch_data_targets = chunked_batch_data
        
        self.counter = 0
        
    
    def __len__(self):
        """used to get loss normalization factor for gradient accumulation
        """
        return self.batch_size
        
    
    def __iter__(self):
        """
        iterating on this object generates dicts unpackable for model and loss
        """        
        output = {'model_inputs': {},
                  'targets': {}}
        
        try:
            for key in self.chunked_batch_data_inputs:
                output['model_inputs'][key] = self.chunked_batch_data_inputs[key][self.counter]
            for key in self.chunked_batch_data_targets:
                output['targets'][key] = self.chunked_batch_data_targets[key][self.counter]
        
        except IndexError:
            raise StopIteration()
        
        self.counter += 1
        
        yield output        
                
                
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
        
        rolling_reward = t.tensor([ep_data['reward'] for ep_data in episode_data], device=device)
        
        target_mask = (starting_points_padded == PADDING_LM_ID).to(device)
        
        targets_idx = t.tensor([ep_data['action_index'] for ep_data in episode_data], device=device)
        
        batch_size = targets_idx.size(0)
        
        chunked_batch_data_inputs = {'pieces_ids': pieces_ids.chunk(nb_chunks),
                                     'colors_ids': colors_ids.chunk(nb_chunks),
                                     'start_move_indexes': starting_points_padded.chunk(nb_chunks),
                                     'end_move_indexes': destinations_padded.chunk(nb_chunks),
                                     'target_mask': target_mask.chunk(nb_chunks)}

        chunked_batch_data_targets =  {'rolling_rewards': rolling_reward.chunk(nb_chunks),
                                       'action_index': targets_idx.chunk(nb_chunks)}
        
        return (chunked_batch_data_inputs, chunked_batch_data_targets), batch_size