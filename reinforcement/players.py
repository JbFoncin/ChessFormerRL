from abc import ABC
from random import choice, random

import torch as t
from chess import BLACK, WHITE

from modeling.tools import prepare_for_model_inference

WHITE_COLOR_MAPPING = {None: 0, 'w': 1, 'b': 2}
BLACK_COLOR_MAPPING = {None: 0, 'b': 1, 'w': 2}

class PlayerABC(ABC):
    def __init__(self, *args, **kwargs):
        self.color_map = None
        self.color = None
    
    def choose_random_action(self, board):
        """
        Args:
            board (chess.Board): the current game state

        Returns:
            str: chosen action, like 'e2e4'
            int: action index
            None: the inference data
        """
        actions = self.get_possible_actions(board)
        action_index = int(random() * len(actions))
        return actions[action_index], action_index, None
        
    @staticmethod
    def get_possible_actions(board):
        """
        Args:
            board (chess.Board): the current game state

        Returns:
            list[str]: list of possible actions, like 'e2e4'
        """
        return [str(move) for move in board.legal_moves]
    
    def set_color(self, color):
        """set player color

        Args:
            color (bool): True for white, False for black
        """
        if color:
            self.color_map = WHITE_COLOR_MAPPING
            self.color = WHITE
        else:
            self.color_map = BLACK_COLOR_MAPPING
            self.color = BLACK


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
            int: action index
        """
        return self.choose_random_action(board)
    

class ModelPlayer(PlayerABC):
    def __init__(self, model, random_action_rate, model_device):
        """
        Args:
            color (str): 'b' for black or 'w' for white
            model (nn.Module): the policy
            
        """
        super().__init__()
        
        self.model = model
        self.random_action_rate = random_action_rate
        self.model_device = model_device

    @t.no_grad()
    def choose_action(self, board):
        """
        Args:
            board (chess.board): the current game object
        Returns:
            str: initital and final positions as str (like 'e2e4')
            int: action index
            inference data: dict of tensor to be stored in the replay buffer
        """
        if random() < self.random_action_rate:
            
            return self.choose_random_action(board)        
        
        possible_actions = self.get_possible_actions(board)
                            
        inference_data = prepare_for_model_inference(board, self.color_map, self.model_device)
        
        actions_scores = self.model(**inference_data)
        
        actions_scores = actions_scores.squeeze(1)
        
        chosen_action_index = t.argmax(actions_scores).cpu().item()
        
        return possible_actions[chosen_action_index], chosen_action_index, inference_data
    
    def choose_random_action(self, board):
        """
        Args:
            board (chess.Board): the current game

        Returns:
            str: next move as str
            int: index of the chosen action
            inference data: dict of tensor to be stored in the replay buffer
        """        
        action, chosen_action_index, _ = super().choose_random_action(board)
        
        inference_data = prepare_for_model_inference(board, self.color_map)
        
        return action, chosen_action_index, inference_data