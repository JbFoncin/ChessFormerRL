import torch as t
from torch.nn.utils.rnn import pad_sequence

from chesstools.tools import (PADDING_LM_ID, get_all_encoded_pieces_and_colors,
                              get_index)


def prepare_input_for_batch(inference_data_list, device='cpu'):
    """generates input for training from previously generated data

    Args:
        inference_data_list (list[dict]): the list of dict to aggregate
        device (str): the device to create the tensors on

    Returns:
        dict: model inputs and targets
    """
    pieces_ids = t.cat([inf_data['pieces_ids'] for inf_data in inference_data_list], dim=0).to(device)
    colors_ids = t.cat([inf_data['colors_ids'] for inf_data in inference_data_list], dim=0).to(device)
    
    starting_points = [inf_data['start_move_indexes'].squeeze(0) for inf_data in inference_data_list]
    starting_points_padded = pad_sequence(starting_points, batch_first=True, padding_value=PADDING_LM_ID).to(device)
    
    destinations = [inf_data['end_move_indexes'].squeeze(0) for inf_data in inference_data_list]
    destinations_padded = pad_sequence(destinations, batch_first=True, padding_value=PADDING_LM_ID).to(device)
    
    attention_mask = make_attention_mask(starting_points_padded).to(device)
    
    targets = t.tensor([inf_data['target'] for inf_data in inference_data_list], device=device)
    targets_idx = t.tensor([inf_data['target_idx'] for inf_data in inference_data_list], device=device)
    
    target_mask = (starting_points_padded == PADDING_LM_ID).to(device)
    
    batch_data = {
        'model_inputs':
            {'pieces_ids': pieces_ids,
             'colors_ids': colors_ids,
             'start_move_indexes': starting_points_padded,
             'end_move_indexes': destinations_padded,
             'decoder_attention_mask': attention_mask,
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
        device (str): the device to create the tensors on
        
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
    
def make_attention_mask(padded_sequence):
    """
    makes attention mask for padded sequences

    Args:
        padded_sequence (t.tensor): the sequence padded

    Returns:
        t.tensor: the mask to be applied after softmax
    """
    
    pad_mask = padded_sequence == PADDING_LM_ID
    
    attention_masks = t.repeat_interleave(pad_mask.unsqueeze(2), 64, dim=2)
    
    return attention_masks

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