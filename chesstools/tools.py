"""
This module contains tools used to convert states and actions from board to model inputs
"""

COORD_MAP = {key: value for key, value in zip('abcdefgh', [0, 1, 2, 3, 4, 5, 6, 7])}
REVERSED_COORD_MAP = {value: key for key, value in COORD_MAP.items()}
PIECES_MAP = {value: i for i, value in enumerate(['b', 'k', 'n', 'p', 'q', 'r', '.'])}
PADDING_LM_ID = 64 # legal move padding value

BOARD_INDEXES = []
for i in range(7, -1, -1):
    BOARD_INDEXES.extend((range(i * 8, (i + 1) * 8)))


def get_piece(coord, board):
    """
    Returns piece type for a given index.
    used to get the piece involved in a possible action.
    This feature may not be useful, see after experiments

    Args:
        coord (str): the piece coordinates from a1 to h8
        board (chess.Board): object managing state and possible actions

    Returns:
        tuple[int, int, int]: piece encoded, color and index
    """
    board_pieces_flattened = str(board).replace('\n', ' ').split(' ')
    index = COORD_MAP[coord[0]] + (8 - (int(coord[1]))) * 8
    piece = board_pieces_flattened[index]
    return piece.lower()


def get_index(coord):
    """
    converts coordinates (str) to int between 0 and 63

    Args:
        coord (str): the piece coordinates from a1 to h8

    Returns:
        int: index on the checkboard
    """
    return COORD_MAP[coord[0]] + (8 - (int(coord[1]))) * 8


def get_all_encoded_pieces_and_colors(board, color_map):
    """
    encodes the whole game state in a list of tuples

    Args:
        board (chess.Board): object managing state and possible actions
        color_map (dict): mapping from colors to model embeddings

    Returns:
        list[tuple[int, str or None]]: each tuple is the id of a piece and its color
    """
    board_pieces_flattened = str(board).replace('\n', ' ').split(' ')
    pieces, colors = [], []
    for box in board_pieces_flattened:
        if box == '.':
            pieces.append(PIECES_MAP[box])
            colors.append(color_map[None])
        elif box.islower():
            pieces.append(PIECES_MAP[box])
            colors.append(color_map['b'])
        else:
            pieces.append(PIECES_MAP[box.lower()])
            colors.append(color_map['w'])

    return pieces, colors


def get_possible_actions(board):
    """
    get all possible actions except pawn transformation forced to queen
    
    Args:
        board (chess.Board): the current game state

    Returns:
        list[str]: list of possible actions, like 'e2e4'
    """
    
    all_possible_actions = [str(move) for move in board.legal_moves]
    
    filtered_actions = []
    
    for action in all_possible_actions:
        if action[-1].isdigit() or action[-1] == 'q':
            filtered_actions.append(action)            
    
    return filtered_actions