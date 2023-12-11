"""
New training paradigm, the model plays against himself and learns with both
point of view.

The objective is to improve stability of the training and double the sample efficiency.
"""

import torch as t
from chess import Board
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from modeling.tools.a2c import A2CChunkedBatchGenerator
from modeling.tools.shared import move_data_to_device
from reinforcement.players import A2CModelPlayer
from reinforcement.reward import get_endgame_reward, get_move_reward


class A2CMirrorTrainer:
    """
    Trainer for Advantage Actor Critic
    """
    def __init__(self, model, optimizer, max_batch_size,
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
        
        self.competitor = A2CModelPlayer(model=model, model_device=model_device)
        self.agent = A2CModelPlayer(model=model, model_device=model_device)
        
        self.competitor.set_color(False)
        self.agent.set_color(True)
        
        self.value_loss_criterion = nn.MSELoss(reduction='sum')
        
        self.current_episode_data = {True: [], False: []}
        
        self.summary_writer = SummaryWriter(f'runs/{experiment_name}')
        
        
    def update_episode_data(self,
                            color,
                            model_inputs,
                            current_action_index,
                            action_policy_score,
                            estimated_state_value,
                            current_reward):
        """
        updates previous state target with the sampled data

        Args:
            color (bool): both colors have separated history data, True for white, False for black
            model_inputs (dict[str, torch.tensor]): model inputs
            current_action_index (int): index of chosen action
            action_policy_score (float): the output of softmax for current action
            estimated_state_value (float): the softmax output associated with current action
            current_reward (float): reward associated with current state and current action
        """
        move_data_to_device(model_inputs, 'cpu')

        self.current_episode_data[color].append({**model_inputs,
                                                 'reward': current_reward,
                                                 'action_index': current_action_index,
                                                 'action_policy_score': action_policy_score,
                                                 'estimated_state_value': estimated_state_value})
        
    
    def update_last_reward(self, color, reward):
        """
        Args:
            color (bool): the key to the queue being updated
            reward (float): the value to be added to the last reward
        """
        if self.current_episode_data[color]:
            self.current_episode_data[color][-1]['reward'] += reward
        
        
    def compute_td_target(self):
        """
        computes target for state value for both players
        """
        for _, episode_data in self.current_episode_data.items():
            
            for step, next_step in zip(episode_data[:-1],
                                       episode_data[1:]):
                
                step['state_value_target'] = step['reward'] + next_step['estimated_state_value']
            
            episode_data[-1]['state_value_target'] = episode_data[-1]['reward']
            
            
    def generate_sample(self, board):
        """
        Each player chooses an action, reward is computed and data added to buffer

        Args:
            board (chess.Board): the current game

        Returns:
            chess.Board: the current game
            bool: True if game is finished
        """
        #agent plays, always first as he plays white
        agent_output = self.agent.choose_action(board)
        #get reward associated with agent move
        agent_reward = get_move_reward(board, agent_output.action)
        #reward of the agent is the negative of competitor reward
        competitor_reward = - agent_reward 
        #change state
        board.push_san(agent_output.action)

        # Check if competitor can play and get reward

        agent_endgame_reward, _ = get_endgame_reward(board, self.competitor.color)

        if agent_endgame_reward is not None: #finish game
            #update reward of agent
            agent_reward += agent_endgame_reward
            #update competitor reward
            competitor_reward -= agent_endgame_reward
            
            game_continues = False
            
        else:
            
            game_continues = True
            
        self.update_episode_data(self.agent.color,
                                 agent_output.inference_data,
                                 agent_output.action_index,
                                 agent_output.policy_score,
                                 agent_output.estimated_state_value,
                                 agent_reward)
        #updates reward associated with last action to opponent buffer
        self.update_last_reward(self.competitor.color,
                                competitor_reward)
        
        if not game_continues:
            
            return board, False

        #mirror opponent plays
        competitor_output = self.competitor.choose_action(board)
        #mirror opponent has a new reward
        competitor_reward = get_move_reward(board, competitor_output.action)
        #to update last agent reward
        agent_reward = - competitor_reward

        board.push_san(competitor_output.action)

        # check if the game is finished after competitor's action

        endgame_reward, _ = get_endgame_reward(board, self.agent.color)

        if endgame_reward is not None:
            
            competitor_reward += endgame_reward
            
            agent_reward -= endgame_reward
            
            game_continues = False
            
        else:
            
            game_continues = True
            
        self.update_episode_data(self.competitor.color,
                                 competitor_output.inference_data,
                                 competitor_output.action_index,
                                 competitor_output.policy_score,
                                 competitor_output.estimated_state_value,
                                 competitor_reward)
        
        self.update_last_reward(self.agent.color, agent_reward)

        return board, game_continues
    
    
    def train_on_history(self):
        """
        updates weights of the Actor-Critic model
        """
        self.optimizer.zero_grad()
        
        episode_data = self.current_episode_data[True] + self.current_episode_data[False]
        
        batch_iterator = A2CChunkedBatchGenerator(episode_data,
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
            
            advantage = targets['state_value_target'] - state_values.detach()
            
            policy_loss = -(t.log(policy_best_scores) * advantage).sum()
            
            value_loss = self.value_loss_criterion(state_values, targets['state_value_target']) 
            
            chunk_loss = (policy_loss + value_loss) / batch_size
            
            chunk_loss.backward()
            
            total_loss += chunk_loss.detach().item()
            
        self.optimizer.step()
        
        self.current_episode_data = {True: [], False: []}
        
        return total_loss
    
    
    def train(self, num_games):
        """
        The training loop.
        
        Args:
            num_games (int): how much games the model will be trained on
        """
        step = 0

        for epoch in tqdm(range(num_games)):

            board = Board()

            game_continues = True

            while game_continues:

                step += 1

                board, game_continues = self.generate_sample(board)
                
            self.compute_td_target()
            
            loss = self.train_on_history()
            
            self.summary_writer.add_scalar('A2C loss', loss, epoch)