"""
Here we try a sequential learning method.
The trainer trains a first model, used to train a better one and so on
"""
from collections import deque
from copy import deepcopy
from random import choices, shuffle

import torch as t
from chess import Board
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from modeling.tools.shared import move_data_to_device
from modeling.tools.dqn import prepare_input_for_dqn_batch
from reinforcement.players import DQNModelPlayer
from reinforcement.reward import get_endgame_reward, get_move_reward


class DQNTrainer:
    """
    Vanilla DQN
    """
    def __init__(self, model, random_action_rate, buffer_size,
                 update_target_q_step, competitor, batch_size,
                 optimizer, experiment_name, models_device, warm_up_steps):
        """
        Args:
            model (t.nn.Module): The model to be trained
            random_action_rate (float): rate of random decisions
            buffer_size (int): maximum history len
            update_target_q_step (int): number of step between q_hat network updates
            competitor (players.PlayerABC): an instance of a derived class of PlayerABC
            batch_size (int): batch size used for model training
            optimizer (t.nn.Optimizer): the optimizer used for the model
            experiment_name (str): name of the experiment, used for tensorboard
            models_device (str): device to be used for target inference and training
            warm_up_steps (int): minimum history len before training
        """
        self.model = model
        self.models_device = models_device
        self._set_frozen_model(model)
        self.batch_size = batch_size
        self.loss_criterion = nn.MSELoss().to(models_device)
        self.optimizer = optimizer

        self.update_target_q_step = update_target_q_step
        self.warm_up_steps = max(warm_up_steps, batch_size)

        self.buffer = deque(maxlen=buffer_size)
        self.previous_action_data = None

        self.competitor = competitor
        
        self.agent = DQNModelPlayer(model=self.model,
                                    random_action_rate=random_action_rate,
                                    model_device=models_device)
        
        self.summary_writer = SummaryWriter(f'runs/{experiment_name}')

    def update_action_data_buffer(self, model_inputs, current_action, current_reward):
        """
        updates previous state target with the maximum q_hat value

        Args:
            q_hat_max (float): _description_
            model_inputs (dict[str, torch.Tensor]): model inputs
            reward (float): reward associated with current state and action
        """
        move_data_to_device(model_inputs, 'cpu')

        if self.previous_action_data is not None:

            self.previous_action_data['q_hat_input'] = model_inputs
            self.buffer.append(self.previous_action_data)

        self.previous_action_data = {**model_inputs,
                                     'reward': current_reward,
                                     'target_idx': current_action}

    def _set_frozen_model(self, model):
        """
        Set or update q_hat model

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

        #agent plays
        player_output = self.agent.choose_action(board)

        reward = get_move_reward(board, player_output.action)

        board.push_san(player_output.action)

        # Check if competitor can play and get reward

        endgame_reward, _ = get_endgame_reward(board, self.competitor.color)

        if endgame_reward is not None: #finish game

            reward += endgame_reward

            self.update_action_data_buffer(player_output.inference_data,
                                           player_output.action_index,
                                           reward)

            self.clean_previous_action_data()

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
                                           reward)

            self.clean_previous_action_data()

            return reward, board, False

        self.update_action_data_buffer(player_output.inference_data,
                                       player_output.action_index,
                                       reward)

        return reward, board, True

    def train(self, num_games):
        """
        Args:
            num_games (int): number of games to train on
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

                if len(self.buffer) > self.warm_up_steps:
                    loss = self.train_batch()
                    self.summary_writer.add_scalar('MSE', loss, step)

                if step % self.update_target_q_step == 0:
                    self._set_frozen_model(self.model)

            self.summary_writer.add_scalar('Total game rewards', game_reward, epoch)
            

    def train_batch(self):
        """
        samples and train one batch

        Returns:
            float: computed loss over the sampled data
        """
        self.optimizer.zero_grad()

        batch = self.make_training_batch()

        model_output = self.model(**batch['model_inputs'])

        predicted = t.gather(model_output, dim=1,
                             index=batch['targets']['targets_idx'].unsqueeze(1))

        loss = self.loss_criterion(predicted.squeeze(-1), batch['targets']['targets'])

        loss.backward()

        self.optimizer.step()

        return loss.cpu().detach().item()
    

    def make_training_batch(self):
        """
        creates batch for training

        Returns:
            dict[str, torch.Tensor]: model inputs and target
        """
        inference_data_list = choices(self.buffer, k=self.batch_size)

        need_update, others = [], []

        for data in inference_data_list:
            if 'q_hat_input' in data:
                need_update.append(data)
            else:
                others.append(data)

        q_hat_batch = prepare_input_for_dqn_batch(need_update,
                                                  device=self.models_device,
                                                  with_target=False)

        max_q_hat, _ = self.frozen_model(**q_hat_batch['model_inputs']).max(1)

        max_q_hat = max_q_hat.to('cpu').detach()

        for i, data in enumerate(need_update):
            data['target'] = max_q_hat[i].item() + data['reward']

        for data in others:
            data['target'] = data['reward']

        batch_data = need_update + others

        batch = prepare_input_for_dqn_batch(batch_data, self.models_device)

        return batch
