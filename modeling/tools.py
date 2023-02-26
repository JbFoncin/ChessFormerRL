import torch as t
from torch.nn.utils.rnn import pad_sequence

from chesstools.tools import (PADDING_LM_ID, get_all_encoded_pieces_and_colors,
                              get_index)


def prepare_input_for_batch(inference_data_list):
    """generates input for training from previously generated data

    Args:
        inference_data_list (list[dict]): the list of dict to aggregate

    Returns:
        dict: model inputs and targets
    """
    pieces_ids = t.cat([inf_data['pieces_ids'] for inf_data in inference_data_list], dim=0)
    colors_ids = t.cat([inf_data['colors_ids'] for inf_data in inference_data_list], dim=0)
    
    starting_points = [inf_data['start_move_indexes'].squeeze(0) for inf_data in inference_data_list]
    starting_points_padded = pad_sequence(starting_points, batch_first=True, padding_value=PADDING_LM_ID)
    
    destinations = [inf_data['end_move_indexes'].squeeze(0) for inf_data in inference_data_list]
    destinations_padded = pad_sequence(destinations, batch_first=True, padding_value=PADDING_LM_ID)
    
    attention_mask = make_attention_mask(starting_points_padded)
    
    targets = t.tensor([inf_data['target'] for inf_data in inference_data_list])
    targets_idx = t.tensor([inf_data['target_idx'] for inf_data in inference_data_list])
    
    target_mask = starting_points_padded == PADDING_LM_ID
    
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

def prepare_for_model_inference(board, color_map):
    """
    Args:
        board (chess.Board): object managing state and possible actions
        color_map (dict): pieces color mapping for the current player
    """
    pieces, colors = get_all_encoded_pieces_and_colors(board, color_map)
    
    pieces = t.tensor(pieces).unsqueeze(0)
    colors = t.tensor(colors).unsqueeze(0)
    possible_actions = [str(move) for move in board.legal_moves]
        
    starting_moves_indexes = [get_index(move[:2]) for move in possible_actions]
    starting_moves_indexes = t.tensor(starting_moves_indexes).unsqueeze(0)
    
    moves_destinations = [get_index(move[2:]) for move in possible_actions]
    moves_destinations = t.tensor(moves_destinations).unsqueeze(0)
    
    inference_data = {
        'pieces_ids': pieces,
        'colors_ids': colors,
        'start_move_indexes': starting_moves_indexes,
        'end_move_indexes': moves_destinations
    }

    return inference_data
    
def make_attention_mask(padded_sequence):
    """makes attention mask for padded sequences

    Args:
        padded_sequence (t.tensor): the sequence padded

    Returns:
        t.tensor: the mask to be applied after softmax
    """
    
    pad_mask = padded_sequence == PADDING_LM_ID
    
    attention_masks = t.repeat_interleave(pad_mask.unsqueeze(2), 64, dim=2)
    
    return attention_masks
