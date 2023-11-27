from random import shuffle
from copy import deepcopy

import numpy as np
import torch as t

from torch import nn
from chess import Board
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from modeling.tools.policy_gradient import PolicyGradientChunkedBatchGenerator
from modeling.tools.shared import move_data_to_device 
from reinforcement.players import A2CModelPlayer
from reinforcement.reward import get_endgame_reward, get_move_reward


class A2CTrainer:
    """
    Trainer for Advantage Actor Critic
    """
    def __init__(self, model, optimizer, competitor, max_batch_size,
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
        self.max_batch_size = max_batch_size
        
        self.competitor = competitor
        self.agent = A2CModelPlayer(model=model, model_device=model_device)
        
        self.value_loss_criterion = nn.MSELoss(reduction='sum')
        
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
    def update_episode_data(self,
                            model_inputs,
                            current_action_index,
                            action_policy_score,
                            estimated_state_value,
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

        self.current_episode_data.append({**model_inputs,
                                          'reward': current_reward,
                                          'action_index': current_action_index,
                                          'action_policy_score': action_policy_score,
                                          'estimated_state_value': estimated_state_value})
        

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

            self.update_episode_data(player_output.inference_data,
                                     player_output.action_index,
                                     player_output.policy_score,
                                     player_output.estimated_state_value,
                                     reward)

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

            self.update_episode_data(player_output.inference_data,
                                     player_output.action_index,
                                     player_output.policy_score,
                                     player_output.estimated_state_value,
                                     reward)

            return reward, board, False

        self.update_episode_data(player_output.inference_data,
                                 player_output.action_index,
                                 player_output.policy_score,
                                 player_output.estimated_state_value,
                                 reward)

        return reward, board, True
    
    
    def train_episode(self):
        """updates weights of the Actor-Critic model
        """
        self.optimizer.zero_grad()
        
        batch_iterator = PolicyGradientChunkedBatchGenerator(self.current_episode_data,
                                                             self.max_batch_size,
                                                             device=self.model_device)
        
        batch_size = len(batch_iterator)
        
        self.model.train()
        
        total_loss = 0
        
        for chunk in batch_iterator:
            
            model_inputs = chunk['model_inputs']
            targets = chunk['targets']
            
            state_values, policy_scores = self.model(**model_inputs)
            
            policy_best_scores, _ = policy_scores.max(axis=1)
            
            advantage = targets['rolling_rewards'] - state_values.detach()
            
            policy_loss = -(t.log(policy_best_scores) * advantage).sum()
            
            value_loss = self.value_loss_criterion(state_values, targets['rolling_rewards']) 
            
            total_loss += (policy_loss + value_loss) / batch_size
            
            total_loss.backward()
            
        self.optimizer.step()
        
        self.current_episode_data = []
        
        return total_loss.item()
    
    
    def train(self, num_games):
        """
        The training loop.
        
        Args:
            num_games (int): how much games the model will be trained on
        """
        step = 0

        for epoch in tqdm(range(num_games)):

            game_reward = 0

            board = self.init_game()

            game_continues = True

            while game_continues:

                step += 1

                reward, board, game_continues = self.generate_sample(board)

                game_reward += reward
                
            loss = self.train_episode()
            
            self.summary_writer.add_scalar('A2C loss', loss, epoch)

            self.summary_writer.add_scalar('Total game rewards', game_reward, epoch)    
    
    