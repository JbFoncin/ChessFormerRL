from abc import ABC
from collections import namedtuple
from random import random

import torch as t
from chess import BLACK, WHITE

from modeling.tools import prepare_for_model_inference

WHITE_COLOR_MAPPING = {None: 0, 'w': 1, 'b': 2}
BLACK_COLOR_MAPPING = {None: 0, 'b': 1, 'w': 2}

PlayerOutput = namedtuple('PlayerOutput',
                          field_names=['action',
                                       'action_index',
                                       'inference_data',
                                       'estimated_action_value',
                                       'model_used'],
                          defaults=[None,
                                    None,
                                    None,
                                    None,
                                    None])

class PlayerABC(ABC):
    def __init__(self, *args, **kwargs):
        self.color_map = None
        self.color = None

    def choose_random_action(self, board):
        """
        Args:
            board (chess.Board): the current game state

        Returns:
            PlayerOutput
        """
        actions = self.get_possible_actions(board)
        action_index = int(random() * len(actions))
        output = PlayerOutput(action=actions[action_index],
                              action_index=action_index)
        return output

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
    """
    player using a model as policy
    """
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

        self.model.eval()
        actions_scores = self.model(**inference_data)
        self.model.train()

        actions_scores = actions_scores.squeeze(1)

        chosen_action_value, chosen_action_index = actions_scores.cpu().topk(1)

        output = PlayerOutput(action=possible_actions[chosen_action_index.item()],
                              action_index=chosen_action_index.item(),
                              estimated_action_value=chosen_action_value.item(),
                              inference_data=inference_data)

        return output

    @t.no_grad()
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

        inference_data = prepare_for_model_inference(board, self.color_map, self.model_device)

        self.model.eval()
        actions_scores = self.model(**inference_data)
        self.model.train()

        output = PlayerOutput(action=action,
                              action_index=chosen_action_index,
                              estimated_action_value=actions_scores[0, chosen_action_index].cpu().item(),
                              inference_data=inference_data)

        return action, chosen_action_index, inference_data


class DoubleModelPlayer(PlayerABC):
    """
    player using a two model as policy randomly
    """
    def __init__(self, model_1, model_2, random_action_rate, model_device):
        """
        Args:
            color (str): 'b' for black or 'w' for white
            model_1 (nn.Module): the first policy
            model_1 (nn.Module): the second policy
            random_action_rate (float): how often the player will act randomly
            model_device (str): device of the models
        """
        super().__init__()

        self.model_1 = model_1
        self.model_2 = model_2
        self.random_action_rate = random_action_rate
        self.model_device = model_device
        self.model = model_1
        self.model_name = 'model_1'

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

        self.model.eval()
        actions_scores = self.model(**inference_data)
        self.model.train()

        actions_scores = actions_scores.squeeze(1)

        chosen_action_value, chosen_action_index = actions_scores.cpu().topk(1)

        output = PlayerOutput(action=possible_actions[chosen_action_index.item()],
                              action_index=chosen_action_index.item(),
                              estimated_action_value=chosen_action_value.item(),
                              inference_data=inference_data,
                              model_used=self.model_name)

        return output

    def switch_models(self):
        """
        changes model
        """
        if self.model is self.model_1:
            self.model = self.model_2
            self.model_name = 'model_2'

        else:
            self.model = self.model_1
            self.model_name = 'model_1'

    def choose_random_action(self, board):
        """
        Args:
            board (chess.Board): the current game

        Returns:
            str: next move as str
            int: index of the chosen action
            inference data: dict of tensor to be stored in the replay buffer
        """
        raise NotImplementedError('Not done yet')
