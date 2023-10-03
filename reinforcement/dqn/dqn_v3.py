"""
This is the final step in the DQN journey.
We go for a quantile regression Deep Q network
"""
from copy import deepcopy
from math import nan
from random import shuffle

import numpy as np
import torch as t
from chess import Board
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from modeling.tools import move_data_to_device, prepare_input_for_batch
from modeling.qr_loss import QRLoss
from reinforcement.dqn.dqn_v2 import DQNTrainerV2
from reinforcement.players import QRModelPlayer
from reinforcement.reward import get_endgame_reward, get_move_reward


class DQNTrainerV3(DQNTrainerV2):
    """
    same as DQN v2, just adding QR DQN
    """
    def __init__(self, model_1, *args, kappa=0.001):
        super().__init__(model_1, *args, **kwargs)
        self.loss = QRLoss(self.agent.model.nb_quantiles, kappa=kappa)
        
    def _make_agent(self, model, model_device):
        """creates the agent, made this way for derived class

        Args:
            model (t.nn.Module derived): the model to be used to take decisions
            model_device (str): the model device

        Returns:
            _type_: _description_
        """
        agent = QRModelPlayer(model=model,
                              random_action_rate=0.0,
                              model_device=model_device)
        
        return agent
    
    @t.no_grad()
    def update_action_data_buffer(self,
                                  model_inputs,
                                  current_action_index,
                                  estimated_action_value,
                                  current_reward):
        """
        updates previous state target with the maximum q_hat value

        Args:
            model_inputs (dict[str, torch.tensor]): model inputs
            current_action_index (int): index of chosen action
            estimated_action_value (float): Q-value associated to the current action
            current_reward (float): reward associated with current state and current action
        """
        self.clean_action_data_buffer_and_sampling()

        move_data_to_device(model_inputs, 'cpu')

        if len(self.previous_actions_data) == self.nb_steps_reward:

            to_buffer = self.previous_actions_data.pop(0)

            to_buffer['q_hat_input'] = model_inputs

            self.buffer.append(to_buffer)

            model_input_copy = deepcopy(model_inputs)
            move_data_to_device(model_input_copy, self.model_device)

            q_hat = self.target_network(**model_input_copy).cpu()

            idx = t.tensor(current_action_index).unsqueeze(0).unsqueeze(1)

            q_hat_action = t.gather(q_hat, dim=1, index=idx).item()

            sampling_score = self.loss(to_buffer['estimated_action_value'] - to_buffer['reward'] - q_hat_action)
            self.sampling_scores[len(self.buffer) - 1] = sampling_score

        for element in self.previous_actions_data:
            element['reward'] += current_reward

        self.previous_actions_data.append({**model_inputs,
                                           'reward': current_reward,
                                           'target_idx': current_action_index,
                                           'estimated_action_value': estimated_action_value})    