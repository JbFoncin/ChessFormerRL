"""
We continue over the sequential training, but we add multi-step reward, importance
sampling and double dqn
"""
from collections import deque
from copy import deepcopy
from random import choices, shuffle

from scipy.special import softmax
import torch as t
from chess import Board
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from modeling.tools import (prepare_for_model_inference,
                            prepare_input_for_batch,
                            move_data_to_device)
from reinforcement.players import ModelPlayer
from reinforcement.reward import get_endgame_reward, get_move_reward


class DQNTrainer:
    """
    Vanilla DQN
    """
    def __init__(self, model_1, model_2, optimizer_1, optimizer_2, buffer_size,
                 revert_models_nb_steps, competitor, batch_size, experiment_name,
                 models_device, nb_steps_reward):
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
        self.model_1, self.model_2 = model_1,  model_2
        self.optimizer_1, self.optimizer_2 = optimizer_1, optimizer_2
        self.models_device = models_device
        self.batch_size = batch_size
        self.loss_criterion = nn.MSELoss().to(models_device)

        self.revert_models_nb_steps = revert_models_nb_steps
        self.nb_steps_reward = nb_steps_reward

        self.buffer = deque(maxlen=buffer_size)
        self.sampling_score = deque(maxlen=buffer_size)

        self.previous_actions_data = []

        self.competitor = competitor

        self.agent = ModelPlayer(model=self.model_1,
                                 random_action_rate=0.0,
                                 model_device=models_device)

        self.summary_writer = SummaryWriter(f'runs/{experiment_name}')

    def update_action_data_buffer(self, model_inputs, current_action_index, current_reward):
        """
        updates previous state target with the maximum q_hat value

        Args:
            q_hat_max (float): _description_
            model_inputs (dict[str, torch.tensor]): model inputs
            reward (float): reward associated with current state encoded in model inputs and current action
        """

        move_data_to_device(model_inputs, 'cpu')

        if len(self.previous_actions_data) == self.nb_steps_reward:
            to_buffer = self.previous_actions_data.pop(0)
            to_buffer['q_hat_input'] = model_inputs
            self.buffer.append(to_buffer)

        for element in self.previous_actions_data:
            element['reward'] += current_reward

        self.previous_actions_data.append({**model_inputs,
                                           'reward': current_reward,
                                           'target_idx': current_action_index})


    def _revert_models_and_optimizers(self):
        """
        Set or update q_hat

        Args:
            model (torch.nn.Module): a q model
        """
        self.model_1, self.model_2 = self.model_2, self.model_1
        self.optimizer_1, self.optimizer_2 = self.optimizer_2, self.optimizer_1
        self.model_1.train()
        self.model_1.requires_grad_(True)
        self.model_2.eval()
        self.model_2.requires_grad_(False)
        setattr(self.agent, 'model', self.model_1)

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

            action, _, _ = self.competitor.choose_action(board)
            board.push_san(action)

        return board

    def clean_previous_actions_data(self):
        """
        called when episode is finished to avoid adding max q_hat
        to the final reward of the previous game
        """
        self.buffer.extend(self.previous_actions_data)
        self.previous_actions_data = []

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

            self.clean_previous_actions_data()

            return reward, board, False

        competitor_action, _, _ = self.competitor.choose_action(board)

        reward -= get_move_reward(board, competitor_action)

        board.push_san(competitor_action)

        # check if the game is finished after competitor's action

        endgame_reward, neutral = get_endgame_reward(board, self.agent.color)

        if endgame_reward is not None:

            if neutral:
                reward += endgame_reward #stalemate remains bad reward
            else:
                reward -= endgame_reward

            self.update_action_data_buffer(q_hat_max, inference_data, action_idx, reward)

            self.clean_previous_actions_data()

            return reward, board, False

        self.update_action_data_buffer(q_hat_max, inference_data, action_idx, reward)

        return reward, board, True

    def train(self, num_games):
        """
        Args:
            num_games (int):
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

                if len(self.buffer) > self.batch_size:
                    loss = self.train_batch()
                    self.summary_writer.add_scalar('MSE', loss, step)

                if step % self.update_target_q_step == 0:
                    self._set_frozen_model(self.model)

            self.summary_writer.add_scalar('Total game rewards', game_reward, epoch)

    def train_batch(self):
        """
        samples and train one batch

        Returns:
            _type_: _description_
        """
        self.optimizer.zero_grad()

        sample = choices(self.buffer, k=self.batch_size)
        batch = prepare_input_for_batch(sample, device=self.model_device)

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
        sampling_probas =

        inference_data_list = choices(self.buffer, k=self.batch_size)

        need_update, others = [], []

        for data in inference_data_list:
            if 'q_hat_input' in data:
                need_update.append(data)
            else:
                others.append(data)

        q_hat_batch = prepare_input_for_batch(need_update, device=self.model_device, with_target=False)

        max_q_hat, _ = self.model_2(**q_hat_batch['model_inputs']).max(1)

        max_q_hat = max_q_hat.to('cpu').detach()

        for i, data in enumerate(need_update):
            data['target'] = max_q_hat[i].item() + data['reward']

        for data in others:
            data['target'] = data['reward']

        batch_data = need_update + others

        batch = prepare_input_for_batch(batch_data, self.model_device)

        return batch
