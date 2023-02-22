from chesstools.tools import get_piece

REWARDS = {'b': 3, 'n': 3, 'p':1, 'q': 10, 'r': 5, '.': 0}
REWARD_CHECKMATE = -30
REWARD_STALEMATE = -10
REWARD_INSUFFICIENT = -30
REWARD_SEVENTYFIVE = -30
REWARD_FIVEFOLD = -30

def get_endgame_reward(board, color):
    """
    Returns reward if game is finished, else zero.
    To be called BEFORE player action
    Args:
        board (chess.Board): current game board
        color (chess.COLOR): current player color

    Returns:
        int: reward associated with endgame state
    """
    
    if board.is_checkmate():
        return REWARD_CHECKMATE
    
    elif board.is_stalemate():
        return REWARD_STALEMATE
    
    elif board.has_unsufficient_material(color):
        return REWARD_INSUFFICIENT
    
    elif board.is_seventyfive_moves():
        return REWARD_SEVENTYFIVE
    
    elif board.is_fivefold_repetition():
        return REWARD_FIVEFOLD
    
    else:
        return 0
    
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
    return REWARDS[piece]