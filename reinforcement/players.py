from abc import ABC
from random import choice, random

import torch as t
from chess import BLACK, WHITE

from modeling.tools import prepare_for_model_inference

WHITE_COLOR_MAPPING = {None: 0, 'w': 1, 'b': 2}
BLACK_COLOR_MAPPING = {None: 0, 'b': 1, 'w': 2}

class PlayerABC(ABC):
    def __init__(self, color, *args, **kwargs):
        """
        Args:
            color (str): 'w' or 'b'
        """
        if color == 'w':
            self.color_map = WHITE_COLOR_MAPPING
            self.color = WHITE
        elif color == 'b':
            self.color_map = BLACK_COLOR_MAPPING
            self.color = BLACK
    
    def choose_random_action(self, board):
        """
        Args:
            board (chess.Board): the current game state

        Returns:
            str: chosen action, like 'e2e4'
        """
        actions = self.get_possible_actions(board)
        return choice(actions)
        
    @staticmethod
    def get_possible_actions(board):
        """
        Args:
            board (chess.Board): the current game state

        Returns:
            list[str]: list of possible actions, like 'e2e4'
        """
        return [str(move) for move in board.legal_moves]


class DummyPlayer(PlayerABC):
    """
    Always plays randomly
    """       
    def choose_action(self, board):
        """
        Args:
            board (chess.Board): the board
        Returns:
            str: initital and final positions as str (like 'e2e4')
        """
        return self.choose_random_action(board)
    

class ModelPlayer(PlayerABC):
    def __init__(self, color, model, random_action_rate):
        """
        Args:
            color (str): 'b' for black or 'w' for white
            model (nn.Module): the policy
            
        """
        super().__init__(color)
        
        self.model = model
        self.random_action_rate = random_action_rate

    @t.no_grad()
    def choose_action(self, board):
        """
        Args:
            board (chess.board): the current game object
        Returns:
            str: initital and final positions as str (like 'e2e4')
        """
        if random() < self.random_action_rate:
            
            return self.choose_random_action(board)        
        
        possible_actions = self.get_possible_actions(board)
                            
        inference_data = prepare_for_model_inference(board, self.color_map)
        
        actions_scores = self.model(**inference_data)
        
        actions_scores = actions_scores.squeeze(1)
        
        chosen_action_index = t.argmax(actions_scores).item()
        
        return possible_actions[chosen_action_index]