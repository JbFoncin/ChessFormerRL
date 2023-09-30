from chesstools.tools import get_piece

REWARDS = {'b': 3, 'n': 3, 'p':1, 'q': 10, 'r': 5, '.': 0}
REWARD_CHECKMATE = 20
REWARD_STALEMATE = 0
REWARD_INSUFFICIENT = 20
REWARD_SEVENTYFIVE = 0
REWARD_FIVEFOLD = 0
MAX_REWARD = 20


def get_endgame_reward(board, color):
    """
    Returns reward if game is finished, else zero.
    To be called BEFORE player action
    Args:
        board (chess.Board): current game board
        color (chess.COLOR): current player color

    Returns:
        int: reward associated with endgame state
        bool: is endgame neutral
    """
    if board.is_checkmate():
        return REWARD_CHECKMATE / MAX_REWARD, False

    elif board.has_insufficient_material(color):
        return REWARD_INSUFFICIENT / MAX_REWARD, False

    elif board.is_stalemate():
        return REWARD_STALEMATE / MAX_REWARD, True

    elif board.is_seventyfive_moves():
        return REWARD_SEVENTYFIVE / MAX_REWARD, True

    elif board.is_fivefold_repetition():
        return REWARD_FIVEFOLD / MAX_REWARD, True

    else:
        return None, None


def get_move_reward(board, move):
    """
    returns the reward of the move

    Args:
        board (chess.Board): current game board
        move (str): starting and ending position of the chosen action

    Returns:
        int: reward value
    """
    destination = move[2:]
    piece = get_piece(destination, board)
    return REWARDS[piece] / MAX_REWARD