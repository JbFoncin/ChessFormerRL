"""
This module is in WIP state
Here we try a sequential learning method.
The trainer trains a first model, used to train a better one and so on
"""
from collections import deque
from copy import deepcopy
from random import randint, random

import torch as t

from modeling.tools import prepare_for_model_inference


class DQNTrainer():
    def __init__(self, model, random_action_rate, buffer_size,
                 update_target_q_step, random_action_rate_decay,
                 Competitor):
        """
        Args:
            model (t.nn.Module): The model to be trained
            random_action_rate (float): rate of random decisions
            buffer_size (int): maximum history len
            update_target_q_step: number of step between reward network updates
        """
        self.model = model
        self._set_frozen_model(model)
        
        self.random_action_rate = random_action_rate
        self.random_action_rate_decay = random_action_rate_decay
        self.update_target_q_step = update_target_q_step
        
        self.buffer = deque(maxlen=buffer_size)
        
        self.previous_action_data = None 
        self.competitor = competitor
        
    def update_action_data_buffer(max_q_hat, current_batch):
        """
        updates previous state target with the maximum q_hat value

        Args:
            max_q_hat (float): _description_
            action_data (_type_): _description_
            reward (_type_): _description_
        """

        
    def _set_frozen_model(self, model):
        """
        Set or update q_hat

        Args:
            model (torch.nn.Module): a q model
        """
        self.frozen_model = deepcopy(model)
        self.frozen_model.eval()
        self.frozen_model.requires_grad_(False)
        
    @t.no_grad()
    def get_state_and_action(self, board):
        
        model_inputs = prepare_for_model_inference(board)
        
        output = self.model(**model_inputs)
        
        output = output.view(-1)
        
        if random() < self.random_action_rate:
            chosen_action_index = randint(0, output.size(0) - 1)
            
        else:
            _, chosen_action_index = output.topk(1)
            chosen_action_index = chosen_action_index.item()
            
        model_inputs['target_idx'] = chosen_action_index
        
        return model_inputs
        
        
        
    def generate_episode(self, board, player, frozen_player):
        
        if self.previous_action_first_reward is None: #game start
            self.init_game(board, player, frozen_player)
        
    def train(self, num_episodes):
        pass