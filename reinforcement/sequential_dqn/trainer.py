"""
This module is in WIP state
Here we try a sequential learning method.
The trainer trains a first model, used to train a better one and so on
"""
from collections import deque
from modeling.tools import prepare_for_model_inference
from copy import deepcopy


class DQNTrainer():
    def __init__(self, model, random_action_rate, buffer_size,
                 update_target_q_step, random_action_rate_decay):
        """
        Args:
            model (t.nn.Module): The model to be trained
            random_action_rate (float): rate of random decisions
            buffer_size (int): maximum history len
            update_target_q_step: number of step between reward network updates
        """
        self.buffer = deque(maxlen=buffer_size)
        self.random_action_rate = random_action_rate
        self.random_action_rate_decay = random_action_rate_decay
        self.model = model
        self.model_frozen = deepcopy(model)
        self.model_frozen.require_grad = False
        self.update_target_q_step = update_target_q_step
        self.previous_action_data = None 
        self.is_previous_decision_random = False
        
    def init_game(self, board, player, frozen_player):
        
        model_inputs = prepare_for_model_inference(board)
        
        predicted = self.model(**model_inputs)
        
        
        
    def generate_episode(self, board, player, frozen_player):
        
        if self.previous_action_first_reward is None: #game start
            self.init_game(board, player, frozen_player)
        
    def train(self, num_episodes):
        pass