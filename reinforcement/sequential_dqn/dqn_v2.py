"""
We continue over the sequential training, but we add multi-step reward, importance
sampling and double dqn
"""
from collections import deque
from random import choices, shuffle

import torch as t
from chess import Board
from scipy.special import softmax
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from copy import deepcopy

from modeling.tools import move_data_to_device, prepare_input_for_batch
from reinforcement.players import ModelPlayer
from reinforcement.reward import get_endgame_reward, get_move_reward


class DQNTrainerV2:
    """
    double DQN with multi-step reward and importance sampling
    """
    def __init__(self, model_1, model_2, optimizer_1, optimizer_2, buffer_size,
                 revert_models_nb_steps, competitor, batch_size, experiment_name,
                 models_device, nb_steps_reward, epsilon_sampling):
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
        self.sampling_scores = deque(maxlen=buffer_size)
        self.epsilon_sampling = epsilon_sampling

        self.previous_actions_data = []

        self.competitor = competitor

        self.agent = ModelPlayer(model=self.model_1,
                                 random_action_rate=0.0,
                                 model_device=models_device)

        self.summary_writer = SummaryWriter(f'runs/{experiment_name}')

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
            reward (float): reward associated with current state and current action
        """

        move_data_to_device(model_inputs, 'cpu')

        if len(self.previous_actions_data) == self.nb_steps_reward:
            to_buffer = self.previous_actions_data.pop(0)
            to_buffer['q_hat_input'] = model_inputs
            self.buffer.append(to_buffer)
            model_input_copy = deepcopy(model_inputs)
            move_data_to_device(model_input_copy, self.models_device)
            q_hat_max = self.model_2(**model_input_copy).max().cpu().item()
            sampling_score = abs(to_buffer['estimated_action_value'] - to_buffer['reward'] - q_hat_max)
            self.sampling_scores.append(sampling_score)

        for element in self.previous_actions_data:
            element['reward'] += current_reward

        self.previous_actions_data.append({**model_inputs,
                                           'reward': current_reward,
                                           'target_idx': current_action_index,
                                           'estimated_action_value': estimated_action_value})

    def clean_previous_actions_data(self):
        """
        called when episode is finished to avoid adding max q_hat
        to the final reward of the previous game
        """
        for element in self.previous_actions_data:
            sampling_score = abs(element['estimated_action_value'] - element['reward']) + self.epsilon_sampling
            self.sampling_scores.append(sampling_score)
        
        self.buffer.extend(self.previous_actions_data)
        self.previous_actions_data = []

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

            competitor_output = self.competitor.choose_action(board)
            board.push_san(competitor_output.action)

        return board


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

        # Chech if competitor can play and get reward

        endgame_reward, _ = get_endgame_reward(board, self.competitor.color)

        if endgame_reward is not None: #finish game

            reward += endgame_reward

            self.update_action_data_buffer(player_output.inference_data,
                                           player_output.action_index,
                                           player_output.estimated_action_value,
                                           reward)

            self.clean_previous_actions_data()

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
                                           player_output.estimated_action_value,
                                           reward)

            self.clean_previous_actions_data()

            return reward, board, False

        self.update_action_data_buffer(player_output.inference_data,
                                       player_output.action_index,
                                       player_output.estimated_action_value,
                                       reward)

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

                if step % self.revert_models_nb_steps == 0:
                    self._revert_models_and_optimizers()

            self.summary_writer.add_scalar('Total game rewards', game_reward, epoch)

    def train_batch(self):
        """
        samples and train one batch

        Returns:
            float: loss value on the current batch
        """
        self.optimizer_1.zero_grad()

        batch, sample_indexes = self.make_training_batch()

        model_output = self.model_1(**batch['model_inputs'])

        predicted = t.gather(model_output, dim=1,
                             index=batch['targets']['targets_idx'].unsqueeze(1))

        loss = self.loss_criterion(predicted.squeeze(-1), batch['targets']['targets'])

        new_sampling_scores = t.abs(predicted.detach().cpu().squeeze(1) - batch['targets']['targets'].cpu()) + self.epsilon_sampling

        for index, value in zip(sample_indexes, new_sampling_scores):
            self.sampling_scores[index] = value.item()

        loss.backward()

        self.optimizer_1.step()

        return loss.cpu().detach().item()

    def make_training_batch(self):
        """
        creates batch for training

        Returns:
            dict[str, torch.Tensor]: model inputs and target
            list[int]: indexes of sampled data in buffer. Used to update the
                       prioritized buffer sampling probas.
        """
        sampling_probas = softmax(self.sampling_scores)
        indexes = list(range(len(self.buffer)))

        batch_data_indexes = choices(indexes,
                                     k=self.batch_size,
                                     weights=sampling_probas)
        batch_data_list = [self.buffer[i] for i in batch_data_indexes]

        # When we separate the data needing update and the other we loose index position
        # and therefore cannot update the priority buffer.

        need_update, others = [], []
        need_update_indexes, other_indexes = [], []

        for index, data in zip(batch_data_indexes, batch_data_list):

            if 'q_hat_input' in data:
                need_update.append(data)
                need_update_indexes.append(index)

            else:
                others.append(data)
                other_indexes.append(index)

        q_hat_batch = prepare_input_for_batch(need_update,
                                              device=self.models_device,
                                              with_target=False)

        q_hat_output = self.model_2(**q_hat_batch['model_inputs']).detach()

        q_hat_values = t.gather(q_hat_output,
                                dim=1,
                                index=q_hat_batch['targets']['targets_idx'].unsqueeze(1)).cpu()


        for i, data in enumerate(need_update):
            data['target'] = q_hat_values[i].item() + data['reward']

        for data in others:
            data['target'] = data['reward']

        batch_data = need_update + others
        data_indexes = need_update_indexes + other_indexes

        batch = prepare_input_for_batch(batch_data, self.models_device)

        return batch, data_indexes
