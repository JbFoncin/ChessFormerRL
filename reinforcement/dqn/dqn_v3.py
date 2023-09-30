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
from reinforcement.players import ModelPlayer
from reinforcement.reward import get_endgame_reward, get_move_reward


class DQNTrainerV3(DQNTrainerV2):
    """
    same as DQN v2, just adding QR DQN
    """
    def __init__(self, model_1, model_2, optimizer, buffer_size,
                 competitor, batch_size, experiment_name, model_device,
                 nb_steps_reward, warm_up_steps, alpha_sampling, beta_sampling,
                 tau, n_quantiles=100):
        """
        Args:
            model_1 (t.nn.Module): The first DQN model
            model_2 (t.nn.Module): The second DQN model
            optimizer_1 (t.optim.Optimizer): optimizer of model_1
            optimizer_2 (t.optim.Optimizer): optimizer of model_2
            buffer_size (int): maximum history len
            revert_models_nb_steps: nb steps between models switching
            competitor (reinforcement.players.PlayerABC derived classes)
            batch_size (int): number of elements per batch when training
            experiment_name (str): name of the tensorboard run
            models_device (str): device used for models
        """
        self.model, self.target_network = model_1,  model_2
        self.target_network.requires_grad_(False)
        self.target_network.eval()
        self.optimizer = optimizer
        self.model_device = model_device
        self.batch_size = batch_size
        self.loss = QRLoss()

        self.nb_steps_reward = nb_steps_reward
        self.buffer = []
        self.buffer_size = buffer_size
        # To improve performance, we use as sampling score buffer a tensor on GPU
        self.sampling_scores = t.tensor([nan] * buffer_size, device=self.model_device)
        self.alpha_sampling = alpha_sampling
        self.beta_sampling = beta_sampling

        self.previous_actions_data = []

        self.competitor = competitor

        self.agent = ModelPlayer(model=model_1,
                                 random_action_rate=0.0,
                                 model_device=model_device)

        self.tau = tau

        self.warm_up_steps = warm_up_steps

        self.summary_writer = SummaryWriter(f'runs/{experiment_name}')
        