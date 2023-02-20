from random import choice

import torch as t

from chesstools import get_all_encoded_pieces_and_colors, get_index

WHITE_COLOR_MAPPING = {None: 0, 'w': 1, 'b': 2}
BLACK_COLOR_MAPPING = {None: 0, 'b': 1, 'w': 2}


class DummyPlayer:
    def __init__(self):
        pass
    
    def choose_action(self, board):
        """
        Args:
            board (chess.Board): the board
        Returns:
            str: initital and final positions as str (like 'e2e4')
        """
        moves = [str(move) for move in board.legal_moves]
        return choice(moves)
    

class ModelPlayer:
    def __init__(self, color, model, device='cpu'):
        """
        Args:
            color (str): 'b' for black or 'w' for white
            model (nn.Module): the policy
        """
        self.model = model
        self.device = device
        if color == 'w':
            self.color_map = WHITE_COLOR_MAPPING
        elif color == 'b':
            self.color_map = BLACK_COLOR_MAPPING
            
    def choose_action(self, board):
        """
        Args:
            board (chess.board): the current game object
        Returns:
            str: initital and final positions as str (like 'e2e4')
        """        
        pieces, colors = get_all_encoded_pieces_and_colors(board, self.color_map)
        
        pieces = t.tensor(pieces).to(self.device)
        colors = t.tensor(colors).to(self.device)
        
        possible_actions = [str(move) for move in board.legal_moves]
        
        starting_moves_indexes = [get_index(move[:2]) for move in possible_actions]
        starting_moves_indexes = t.tensor(starting_moves_indexes).to(self.device)
        
        moves_destinations = [get_index(move[2:]) for move in possible_actions]
        moves_destinations = t.tensor(moves_destinations).to(self.device)
        
        with t.no_grad():
            actions_scores = self.model(pieces,
                                        colors, 
                                        starting_moves_indexes,
                                        moves_destinations)
        
        actions_scores = actions_scores.to('cpu').squeeze(1).detach()
        
        chosen_action_index = t.argmax(actions_scores)
        
        return possible_actions[int(chosen_action_index)]