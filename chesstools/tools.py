"""
This module contains tools used to convert states and actions from board to model
"""


COORD_MAP = {key: value for key, value in zip('abcdefgh', [0, 1, 2, 3, 4, 5, 6, 7])}
REVERSED_COORD_MAP = {value: key for key, value in COORD_MAP.items()}
PIECES_MAP = {value: i for i, value in enumerate(['b', 'k', 'n', 'p', 'q', 'r', '.'])}


def get_piece_and_index(coord, board):
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
    index = COORD_MAP[coord[0]] * 8 + int(coord[1]) - 1
    piece = board_pieces_flattened[index]
    return PIECES_MAP[piece.lower()], piece.islower(), index

def get_index(coord):
    """
    converts coordinates (str) to int between 0 and 63

    Args:
        coord (str): the piece coordinates from a1 to h8

    Returns:
        int: index on the checkboard
    """
    return COORD_MAP[coord[0]] * 8 + int(coord[1]) - 1

def get_all_encoded_pieces_and_colors(board):
    """
    encodes the whole game state in a list of tuples

    Args:
        board (chess.Board): object managing state and possible actions

    Returns:
        list[tuple[int, str or None]]: each tuple is the id of a piece and its color
    """
    board_pieces_flattened = str(board).replace('\n', ' ').split(' ')
    result = []
    for box in board_pieces_flattened:
        if box == '.':
            result.append((PIECES_MAP[box], None))
        elif box.islower():
            result.append((PIECES_MAP[box], 'b'))
        else:
            result.append((PIECES_MAP[box.lower()], 'w'))
    return result