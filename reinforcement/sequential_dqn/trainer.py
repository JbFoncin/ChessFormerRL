"""
Here we try a sequential learning method.
The trainer trains a first model, used to train a better one and so on
"""
from collections import deque
from copy import deepcopy
from random import shuffle

import torch as t
from chess import Board
from tqdm import tqdm

from modeling.tools import prepare_for_model_inference
from reinforcement.players import ModelPlayer
from reinforcement.reward import get_endgame_reward, get_move_reward


class DQNTrainer():
    """
    Vanilla DQN
    """
    def __init__(self, model, random_action_rate, buffer_size,
                 update_target_q_step, competitor):
        """
        Args:
            model (t.nn.Module): The model to be trained
            random_action_rate (float): rate of random decisions
            buffer_size (int): maximum history len
            update_target_q_step: number of step between q_hat network updates
        """
        self.model = model
        self._set_frozen_model(model)
        
        self.update_target_q_step = update_target_q_step
        
        self.buffer = deque(maxlen=buffer_size)
        
        self.previous_action_data = None 
        
        self.competitor = competitor
        self.agent = ModelPlayer(model=self.model, random_action_rate=random_action_rate)
        
    def update_action_data_buffer(self, q_hat_max, model_inputs, current_action, current_reward):
        """
        updates previous state target with the maximum q_hat value

        Args:
            q_hat_max (float): _description_
            model_inputs (dict[str, torch.tensor]): model inputs
            reward (float): reward associated with current state encoded in model inputs and current action
        """
        if self.previous_action_data is not None:
            
            self.previous_action_data['target'] += q_hat_max
            self.buffer.append(self.previous_action_data)
        
        self.previous_action_data = {**model_inputs, 'target': current_reward, 'target_idx': current_action}
        
    @t.no_grad
    def get_q_hat_max(self, board):
        """
        Args:
            board (chess.Board): the current game

        Returns:
            float: the q_hat_max_value
        """
        data_inference = prepare_for_model_inference(board, self.agent.color_map)
        q_hat_values = self.frozen_model(**data_inference)
        q_hat_max = q_hat_values.max().item()
        
        return q_hat_max
        
    def _set_frozen_model(self, model):
        """
        Set or update q_hat

        Args:
            model (torch.nn.Module): a q model
        """
        self.frozen_model = deepcopy(model)
        self.frozen_model.eval()
        self.frozen_model.requires_grad_(False)
        
    def init_game(self):
        """
        inits board, set players colors, manage first action-state if competitor is white

        Returns:
            chess.Board: the new game
        """
        color_agent, color_competitor = shuffle([True, False]) 
        # True for white and False for black like in the chess package
        self.agent.set_color(color_agent)
        self.competitor.set_color(color_competitor)
        
        board = Board()
        
        if self.competitor.color: #if competitor plays first
            
            action, _ = self.competitor.choose_action(board)
            board.push_san(action)
        
        return board
    
    def clean_previous_action_data(self):
        """
        called when episode is finished to avoid adding max q_hat
        to the final reward of the previous game
        """
        self.buffer.append(self.previous_action_data)
        self.previous_action_data = None

    def generate_sample(self, board):
        """
        each player choose an action, reward is computed and data added to buffer

        Args:
            board (chess.Board): the current game
            
        Returns:
            chess.Board: the current game
            bool: True if game is finished
        """
        
        q_hat_max = self.get_q_hat_max(board)
        
        #agent plays
        action, action_idx, inference_data = self.agent.choose_action(board)            
        
        reward = get_move_reward(board, action)
        
        board.push_san(action)
        
        # Chech if competitor can play and get reward
        
        endgame_reward, _ = get_endgame_reward(board, self.competitor.color)
        
        if endgame_reward is not None: #finish game
            
            reward += endgame_reward
            
            self.update_action_data_buffer(q_hat_max, inference_data, action_idx, reward)
            
            self.clean_previous_action_data()
            
            return reward, board, False
        
        competitor_action, _, _ = self.competitor.choose_action(board)
        
        reward -= get_move_reward(board, action)
        
        board.push_san(competitor_action)
        
        # check if the game is finished after competitor's action
        
        endgame_reward, neutral = get_endgame_reward(board, self.agent.color)
        
        if endgame_reward is not None:
            
            if neutral:
                reward += endgame_reward #stalemate remains bad reward
            else:
                reward -= endgame_reward
            
            self.update_action_data_buffer(q_hat_max, inference_data, action_idx, reward)
            
            self.clean_previous_action_data()
            
            return reward, board, False
        
        self.update_action_data_buffer(q_hat_max, inference_data, action_idx, reward)
        
        return reward, board, True
              
    def train(self, num_games):
        """
        Args:
            num_games (int): 
        """        
        for _ in tqdm(range(num_games)):
            
            game_reward = 0
            
            board = self.init_game()
            
            game_continues = False
            
            while game_continues:
                reward, board, game_continues = self.generate_sample(board)
                game_reward += reward