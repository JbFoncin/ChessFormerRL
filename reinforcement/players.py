from abc import ABC
from random import choice
import torch as t

from chesstools import get_all_encoded_pieces_and_colors

WHITE_COLOR_MAPPING = {None: 0, 'w': 1, 'b': 2}
BLACK_COLOR_MAPPING = {None: 0, 'b': 1, 'w': 2}


class DummyPlayer:
    def __init__(self):
        pass
    
    def choose_action(self, board):
        """
        Args:
            board (chess.Board): the board
        """
        moves = [str(move) for move in board.legal_moves]
        return choice(moves)
    

class ModelPlayer:
    def __init__(self, color, model, random_action_rate=0.0):
        """
        Args:
            color (str): 'b' for black or 'w' for white
            model (nn.Module): the policy
            random_action_rate (float, optional): _description_. Defaults to 0.25.
        """
        self.model = model
        self.random_action_rate = random_action_rate
        if color == 'w':
            self.color_map = WHITE_COLOR_MAPPING
        elif color == 'b':
            self.color_map = BLACK_COLOR_MAPPING
            
    def choose_action(self, board):
        """
        Args:
            board (chess.board): the current game object
        """
        pieces, colors = get_all_encoded_pieces_and_colors(board, self.color_map)
        pieces = t.tensor(pieces).to(self.model.device)
        colors = t.tensor(colors).to(self.model.device)
        possible_actions = [str(move) for move in board.legal_moves]