import torch as t
from torch.nn.utils.rnn import pad_sequence

from chesstools.tools import (PADDING_LM_ID, get_all_encoded_pieces_and_colors,
                              get_index)


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
        chunked_batch_data = self.prepare_input_for_policy_gradient_batch(episode_data,
                                                                          max_batch_size,
                                                                          device)
        
        self.chunked_batch_data_inputs, self.chunked_batch_data_targets = chunked_batch_data
        
        self.counter = 0
        
    
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
        
        return output        
                
                
    @staticmethod
    def prepare_input_for_policy_gradient_batch(episode_data, max_batch_size, device='cpu'):
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
        
        targets_idx = t.tensor([ep_data['target_idx'] for ep_data in episode_data], device=device)
        
        chunked_batch_data_inputs = {'pieces_ids': pieces_ids.chunk(nb_chunks),
                                    'colors_ids': colors_ids.chunk(nb_chunks),
                                    'start_moves_indexes': starting_points_padded.chunk(nb_chunks),
                                    'end_move_indexes': destinations_padded.chunk(nb_chunks),
                                    'target_mask': target_mask.chunk(nb_chunks)}

        chunked_batch_data_targets =  {'rolling_rewards': rolling_reward.chunk(nb_chunks),
                                       'target_idx': targets_idx.chunk(nb_chunks)}
        
        return chunked_batch_data_inputs, chunked_batch_data_targets


def prepare_input_for_dqn_batch(inference_data_list, device='cpu', with_target=True,
                                quantile_reg=False):
    """
    generates input for training from previously generated data

    Args:
        inference_data_list (list[dict]): the list of dict to aggregate
        device (str): the device to create the tensors on
        with_target (bool): True if target is required
        quantile_reg (bool or int): False if target is a scalar. Else, target is a tensor of quantiles.
                                    In this case
                                    Default to False

    Returns:
        dict: model inputs and targets
    """
    pieces_ids = t.cat([inf_data['pieces_ids'] for inf_data in inference_data_list], dim=0).to(device)
    colors_ids = t.cat([inf_data['colors_ids'] for inf_data in inference_data_list], dim=0).to(device)

    starting_points = [inf_data['start_move_indexes'].squeeze(0) for inf_data in inference_data_list]
    starting_points_padded = pad_sequence(starting_points, batch_first=True, padding_value=PADDING_LM_ID).to(device)

    destinations = [inf_data['end_move_indexes'].squeeze(0) for inf_data in inference_data_list]
    destinations_padded = pad_sequence(destinations, batch_first=True, padding_value=PADDING_LM_ID).to(device)

    targets_idx = t.tensor([inf_data['target_idx'] for inf_data in inference_data_list], device=device)
    
    if quantile_reg:
        
        batch_size = len(inference_data_list)
        targets_idx = t.repeat_interleave(targets_idx, quantile_reg).view(batch_size, -1).unsqueeze(1)
        
        if with_target:
            targets = t.cat([inf_data['target'].unsqueeze(0) for inf_data in inference_data_list],
                            dim=0).squeeze(1).to(device)
        else:
            targets = None

    else:
        
        if with_target:
            targets = t.tensor([inf_data['target'] for inf_data in inference_data_list], device=device)
        else:
            targets = None

    target_mask = (starting_points_padded == PADDING_LM_ID).to(device)
    
    if quantile_reg:
        target_mask = target_mask.unsqueeze(-1).repeat(1, 1, quantile_reg)

    batch_data = {
        'model_inputs':
            {'pieces_ids': pieces_ids,
             'colors_ids': colors_ids,
             'start_move_indexes': starting_points_padded,
             'end_move_indexes': destinations_padded,
             'target_mask': target_mask},
        'targets':
            {'targets': targets,
             'targets_idx': targets_idx}
    }

    return batch_data




def prepare_for_model_inference(board, color_map, device='cpu'):
    """
    Args:
        board (chess.Board): object managing state and possible actions
        color_map (dict): pieces color mapping for the current player
        device (str): the device to create the tensors on. Default to 'cpu'

    Returns:
        dict[t.Tensor]: dict of tensors needed for model inference
    """
    pieces, colors = get_all_encoded_pieces_and_colors(board, color_map)

    pieces = t.tensor(pieces).unsqueeze(0).to(device)
    colors = t.tensor(colors).unsqueeze(0).to(device)
    possible_actions = [str(move) for move in board.legal_moves]

    starting_moves_indexes = [get_index(move[:2]) for move in possible_actions]
    starting_moves_indexes = t.tensor(starting_moves_indexes, device=device).unsqueeze(0)

    moves_destinations = [get_index(move[2:]) for move in possible_actions]
    moves_destinations = t.tensor(moves_destinations, device=device).unsqueeze(0)

    inference_data = {
        'pieces_ids': pieces,
        'colors_ids': colors,
        'start_move_indexes': starting_moves_indexes,
        'end_move_indexes': moves_destinations
    }

    return inference_data


def move_data_to_device(data, device):
    """
    Moves recursively data to device

    Args:
        data (dict): a dict of dict or tensors
        device (str): 'cuda' or 'cpu'
    """
    for key in data:

        if isinstance(data[key], t.Tensor):
            data[key] = data[key].to(device)

        if isinstance(data[key], dict):
            return move_data_to_device(data[key], device)