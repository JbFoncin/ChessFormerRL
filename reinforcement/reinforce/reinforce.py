"""
First attempt with a policy based strategy
"""
from random import shuffle
from copy import deepcopy

import numpy as np
import torch as t
from chess import Board
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from modeling.tools import move_data_to_device, prepare_input_for_batch
from reinforcement.players import ModelPlayer
from reinforcement.reward import get_endgame_reward, get_move_reward


class ReinforceTrainer:
    """
    Basic reinforce algorithm trainer
    """
    def __init__(self, model, optimizer, competitor, batch_size,
                 experiment_name, model_device):
        """
        Args:
            model (t.nn.Module): the model to be trained
            optimizer (t.optim.Optimizer): model optimizer
            competitor (reinforcement.players.PlayerABC derived classes): the agent opponent
            batch_size (int): number of elements per batch when training. If The episode history length is 
                              bigger than this value, we perform gradient accumulation.
            experiment_name (str): the experiment name, used in tensorboard
            model_device (str): 'cuda' or 'cpu' 
        """
        self.model = model
        self.optimizer = optimizer
        self.model_device = model_device
        self.batch_size = batch_size
        
        self.competitor = competitor
        self.agent = ModelPlayer(model=model, random_action_rate=0.0, model_device=model_device)
        
        self.current_episode_data = []
        
        self.summary_writer = SummaryWriter(f'runs/{experiment_name}')
        
    
    def init_game(self):
        """
        inits board, set players colors, manage first action-state if competitor is white

        Returns:
            chess.Board: the new game
        """
        colors = [True, False]
        shuffle(colors)
        color_agent, color_competitor = colors
        # True for white and False for black like in the chess package
        self.agent.set_color(color_agent)
        self.competitor.set_color(color_competitor)

        board = Board()

        if self.competitor.color: #if competitor plays first

            competitor_output = self.competitor.choose_action(board)
            board.push_san(competitor_output.action)

        return board


    @t.no_grad()
    def update_current_episode_data(self,
                                    model_inputs,
                                    current_action_index,
                                    estimated_action_value,
                                    current_reward):
        """
        updates previous state target with the maximum q_hat value

        Args:
            model_inputs (dict[str, torch.tensor]): model inputs
            current_action_index (int): index of chosen action
            estimated_action_value (float): the softmax output associated with current action
            current_reward (float): reward associated with current state and current action
        """
        move_data_to_device(model_inputs, 'cpu')

        for element in self.current_episode_data:
            element['reward'] += current_reward

        self.self.current_episode_data.append({**model_inputs,
                                               'reward': current_reward,
                                               'target_idx': current_action_index,
                                               'estimated_action_value': estimated_action_value})
 

    def generate_sample(self, board):
        """
        Each player chooses an action, reward is computed and data added to buffer

        Args:
            board (chess.Board): the current game

        Returns:
            chess.Board: the current game
            bool: True if game is finished
        """
        #agent plays
        player_output = self.agent.choose_action(board)

        reward = get_move_reward(board, player_output.action)

        board.push_san(player_output.action)

        # Chech if competitor can play and get reward

        endgame_reward, _ = get_endgame_reward(board, self.competitor.color)

        if endgame_reward is not None: #finish game

            reward += endgame_reward

            self.update_action_data_buffer(player_output.inference_data,
                                           player_output.action_index,
                                           player_output.estimated_action_value,
                                           reward)
            self.clean_previous_actions_data()

            return reward, board, False

        competitor_output = self.competitor.choose_action(board)

        reward -= get_move_reward(board, competitor_output.action)

        board.push_san(competitor_output.action)

        # check if the game is finished after competitor's action

        endgame_reward, neutral = get_endgame_reward(board, self.agent.color)

        if endgame_reward is not None:

            if neutral:
                reward += endgame_reward #stalemate remains bad reward
            else:
                reward -= endgame_reward

            self.update_action_data_buffer(player_output.inference_data,
                                           player_output.action_index,
                                           player_output.estimated_action_value,
                                           reward)

            return reward, board, False

        self.update_action_data_buffer(player_output.inference_data,
                                       player_output.action_index,
                                       player_output.estimated_action_value,
                                       reward)

        return reward, board, True