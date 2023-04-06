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
from reinforcement.players import ModelPlayer
from reinforcement.reward import get_endgame_reward, get_move_reward

