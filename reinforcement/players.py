from abc import ABC
from collections import namedtuple
from random import random

import torch as t
from chess import BLACK, WHITE

from modeling.tools.shared import prepare_for_model_inference

WHITE_COLOR_MAPPING = {None: 0, 'w': 1, 'b': 2}
BLACK_COLOR_MAPPING = {None: 0, 'b': 1, 'w': 2}

#The PlayerOutputDQN and PlayerOutputPG behave the same way,
#the only difference is with the labels as estimated action value can
#be confusing when working with policy gradients
PlayerOutputDQN = namedtuple('PlayerOutputDQN',
                             field_names=['action',
                                          'action_index',
                                          'inference_data',
                                          'estimated_action_value'],
                             defaults=[None,
                                       None,
                                       None,
                                       None])


#PG means PolicyGradient
PlayerOutputPG = namedtuple('PlayerOutput',
                            field_names=['action',
                                         'action_index',
                                         'inference_data',
                                         'policy_score'],
                            defaults=[None,
                                      None,
                                      None,
                                      None])


PlayerOutputA2C = namedtuple('PlayerOutputA2C',
                             field_names=['action',
                                          'action_index',
                                          'inference_data',
                                          'policy_score',
                                          'estimated_state_value'],
                             defaults=[None,
                                       None,
                                       None,
                                       None,
                                       None])


class PlayerABC(ABC):
    """
    abstract baseclass used for all players
    """
    def __init__(self, *args, **kwargs):
        self.color_map = None
        self.color = None

    @staticmethod
    def get_possible_actions(board):
        """
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

    def set_color(self, color):
        """
        set player color

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
    
    def choose_random_action(self, board):
        """
        Args:
            board (chess.Board): the current game state

        Returns:
            PlayerOutput
        """
        actions = self.get_possible_actions(board)
        action_index = int(random() * len(actions))
        output = PlayerOutputDQN(action=actions[action_index],
                                 action_index=action_index)
        return output


class DQNModelPlayer(PlayerABC):
    """
    player using a model as policy, works with DQN or reinforce
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

        actions_scores = self.model(**inference_data)

        actions_scores = actions_scores.squeeze(1)

        chosen_action_value, chosen_action_index = actions_scores.cpu().topk(1)

        output = PlayerOutputDQN(action=possible_actions[chosen_action_index.item()],
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

        actions_scores = self.model(**inference_data)

        output = PlayerOutputDQN(action=action,
                                 action_index=chosen_action_index,
                                 estimated_action_value=actions_scores[0, chosen_action_index].cpu().item(),
                                 inference_data=inference_data)

        return output
    
    
class PolicyGradientModelPlayer(PlayerABC):
    """
    player using a model as policy, works with DQN or reinforce
    """
    def __init__(self, model, model_device):
        """
        Args:
            color (str): 'b' for black or 'w' for white
            model (nn.Module): the policy
        """
        super().__init__()

        self.model = model
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
        possible_actions = self.get_possible_actions(board)

        inference_data = prepare_for_model_inference(board, self.color_map, self.model_device)

        policy_scores = self.model(**inference_data)

        policy_scores = policy_scores.squeeze(1)

        policy_score, chosen_action_index = policy_scores.cpu().topk(1)

        output = PlayerOutputPG(action=possible_actions[chosen_action_index.item()],
                                action_index=chosen_action_index.item(),
                                policy_score=policy_score.item(),
                                inference_data=inference_data)

        return output


class QRDQNModelPlayer(DQNModelPlayer):
    """
    Player using a quantile regression model
    """
    @t.no_grad()
    def choose_action(self, board):
        """
        Args:
            board (chess.board): the current game object
        
        Returns:
            NamedTuple: object contains :
                -The action chosen
                -Its index
                -The estimated quantiles of the chosen action reward
                -Inference data used for forward pass
        """
        if random() < self.random_action_rate:

            return self.choose_random_action(board)

        possible_actions = self.get_possible_actions(board)

        inference_data = prepare_for_model_inference(board, self.color_map, self.model_device)

        actions_scores = self.model(**inference_data)

        _, index = actions_scores.mean(2).max(1)

        value = actions_scores[0, index.item(), :].unsqueeze(0).cpu()

        output = PlayerOutputDQN(action=possible_actions[index.item()],
                                 action_index=index.item(),
                                 estimated_action_value=value,
                                 inference_data=inference_data)

        return output

    @t.no_grad()
    def choose_random_action(self, board):
        """
        Args:
            board (chess.Board): the current game
        
        Returns:
            NamedTuple: object contains :
                -The action chosen
                -Its index
                -The estimated quantiles of the chosen action reward
                -Inference data used for forward pass
        """
        action, chosen_action_index, _ = super().choose_random_action(board)

        inference_data = prepare_for_model_inference(board, self.color_map, self.model_device)

        actions_scores = self.model(**inference_data)
        value = actions_scores[0, chosen_action_index, :].unsqueeze(0).cpu()
        #The action value is now the estimated quantiles of the Q function
        output = PlayerOutputDQN(action=action,
                                 action_index=chosen_action_index,
                                 estimated_action_value=value,
                                 inference_data=inference_data)

        return output
    
    
class A2CModelPlayer(PlayerABC):
    """
    player using a model as policy, works with DQN or reinforce
    """
    def __init__(self, model, model_device):
        """
        Args:
            color (str): 'b' for black or 'w' for white
            model (nn.Module): the policy
        """
        super().__init__()

        self.model = model
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
        possible_actions = self.get_possible_actions(board)

        inference_data = prepare_for_model_inference(board, self.color_map, self.model_device)

        state_value, policy_scores = self.model(**inference_data) #The action values are the advantage from each action

        policy_scores = policy_scores.squeeze(1)

        policy_score, chosen_action_index = policy_scores.cpu().topk(1)

        output = PlayerOutputA2C(action=possible_actions[chosen_action_index.item()],
                                 action_index=chosen_action_index.item(),
                                 policy_score=policy_score.item(),
                                 estimated_state_value=state_value,
                                 inference_data=inference_data)

        return output