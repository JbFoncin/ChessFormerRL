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