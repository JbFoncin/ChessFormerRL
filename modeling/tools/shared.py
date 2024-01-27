import torch as t

from chesstools.tools import (get_all_encoded_pieces_and_colors,
                              get_index,
                              get_possible_actions)


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
    
    possible_actions = get_possible_actions(board)

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
    Moves recursively data to device.

    Args:
        data (dict): a dict of dict or tensors
        device (str): 'cuda' or 'cpu'
    """
    for key in data:

        if isinstance(data[key], t.Tensor):
            data[key] = data[key].to(device)

        if isinstance(data[key], dict):
            return move_data_to_device(data[key], device)
        
        
def compute_entropy(policy_softmax, target_mask):
    """
    Computes entropy from policy decoder output.
    Used to increase environment exploration.

    Args:
        policy_softmax (t.Tensor): tensor with size (batch, action_space), padded with 0
        target_mask (t.Tensor): True if padding action space, False for scores to evaluate
    """    
    target_mask_reversed = ~ target_mask
    
    #we remove each element on batch axis with action space size equal to one as
    #entropy has no meaning for a sequence of one element
    action_space_size_mask = target_mask_reversed.sum(dim=1) > 1
    policy_softmax_filtered = policy_softmax[action_space_size_mask, :]
    
    #We also filter the mask to keep it aligned with softmax output
    target_mask_reversed_filtered = target_mask_reversed[action_space_size_mask, :]
    batch_size, action_space_size = target_mask_reversed_filtered.size()
    
    #used to normalize the entropy scale as entropy increases with action space size
    action_space_size = target_mask_reversed_filtered.sum(dim=1).view(batch_size, 1).repeat(1, action_space_size)
    entropy_scale = action_space_size[target_mask_reversed_filtered]
    
    #We finally compute entropy
    entropy_values = policy_softmax_filtered[target_mask_reversed_filtered]
    entropy = entropy_values * t.log(entropy_values) / t.log(entropy_scale)
    #We average it by batch size
    return entropy.sum()
    