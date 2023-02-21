REWARDS = {'b': 3, 'n': 3, 'p':1, 'q': 10, 'r': 5, '.': 0}
REWARD_CHECKMATE = -30
REWARD_STALEMATE = -10
REWARD_INSUFFICIENT = -30
REWARD_SEVENTYFIVE = -30
REWARD_FIVEFOLD = -30

def get_reward_and_status(board, action, color):
    
    if board.is_checkmate():
        return REWARD_CHECKMATE, False
    
    elif board.is_stalemate():
        return REWARD_STALEMATE, False
    
    elif board.has_unsufficient_material(color):
        return REWARD_INSUFFICIENT, False
    
    elif board.is_seventyfive_moves():
        return REWARD_SEVENTYFIVE
    
    elif board.is_fivefold_repetition():
        return
    else 
