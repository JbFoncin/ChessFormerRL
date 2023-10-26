"""
We continue over the sequential training, but we add multi-step reward, importance
sampling and double dqn
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


class DQNTrainerV2:
    """
    double DQN with multi-step reward and importance sampling
    """
    def __init__(self, model_1, model_2, optimizer, buffer_size,
                 competitor, batch_size, experiment_name, model_device,
                 nb_steps_reward, warm_up_steps, alpha_sampling, beta_sampling,
                 tau):
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
            nb_steps_rewards (int): The number of steps to unroll Bellman equation
                                    for more information see https://arxiv.org/pdf/1703.01327.pdf
            warm_up_steps (int): minimum size of buffer to acquire at the beginning of the train
            alpha_sampling (float): hyperparameter for batch sampling
            beta_sampling (float): hyperparameter for batch weighting
        """
        self.model, self.target_network = model_1,  model_2
        self.target_network.requires_grad_(False)
        self.target_network.eval()
        self.optimizer = optimizer
        self.model_device = model_device
        self.batch_size = batch_size

        self.nb_steps_reward = nb_steps_reward
        self.buffer = []
        self.buffer_size = buffer_size
        # To improve performance, we use as sampling score buffer a tensor on GPU
        self.sampling_scores = t.tensor([nan] * buffer_size, device=self.model_device)
        self.alpha_sampling = alpha_sampling
        self.beta_sampling = beta_sampling

        self.previous_actions_data = []

        self.competitor = competitor

        self.agent = self._make_agent(model_1, model_device=model_device)

        self.tau = tau

        self.warm_up_steps = warm_up_steps

        self.summary_writer = SummaryWriter(f'runs/{experiment_name}')

    def _make_agent(self, model, model_device):
        """creates the agent, made this way for derived class

        Args:
            model (t.nn.Module derived): the model to be used to take decisions
            model_device (str): the model device

        Returns:
            _type_: _description_
        """
        agent = ModelPlayer(model=model,
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

            sampling_score = abs(to_buffer['estimated_action_value'] - to_buffer['reward'] - q_hat_action)
            self.sampling_scores[len(self.buffer) - 1] = sampling_score
            
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
        self.clean_action_data_buffer_and_sampling()

        for element in self.previous_actions_data:
            sampling_score = abs(element['estimated_action_value'] - element['reward'])
            self.buffer.append(element)
            self.sampling_scores[len(self.buffer) - 1] = sampling_score

        self.previous_actions_data = []


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
                if len(self.buffer) >= self.warm_up_steps:
                    loss = self.train_batch()
                    self.summary_writer.add_scalar('MSE', loss, step)

            self.summary_writer.add_scalar('Total game rewards', game_reward, epoch)

    def train_batch(self):
        """
        samples and train one batch

        Returns:
            float: loss value on the current batch
        """
        self.optimizer.zero_grad()

        batch, sample_indexes, weights = self.make_training_batch()

        model_output = self.model(**batch['model_inputs'])

        predicted = t.gather(model_output, dim=1,
                             index=batch['targets']['targets_idx'].unsqueeze(1))

        loss = (predicted.squeeze(-1) - batch['targets']['targets']) ** 2
        loss = (loss * weights).mean()
        loss.backward()

        new_sampling_scores = t.abs(predicted.detach().cpu().squeeze(1) - batch['targets']['targets'].cpu())

        self.optimizer.step()

        for index, value in zip(sample_indexes, new_sampling_scores):
            self.sampling_scores[index] = value.item()

        model_state_dict = self.model.state_dict()
        target_network_state_dict = self.target_network.state_dict()

        for key in target_network_state_dict:
            target_network_state_dict[key] = self.tau * model_state_dict[key] + (1 - self.tau) * target_network_state_dict[key]

        self.target_network.load_state_dict(target_network_state_dict)

        return loss.cpu().detach().item()

    @t.no_grad()
    def make_training_batch(self):
        """
        creates batch for training

        Returns:
            dict[str, torch.Tensor]: model inputs and target
            list[int]: indexes of sampled data in buffer. Used to update the
                       prioritized buffer sampling probas.
            np.array: weights to be used for batch training
        """
        batch_data_indexes, weights = self.sample_indexes()

        batch_data_list = [self.buffer[i] for i in batch_data_indexes]

        # When we separate the data needing update and the other we loose index position
        # and therefore cannot update the priority buffer.

        need_update, others = [], []
        need_update_indexes, other_indexes = [], []
        weights_update, weights_other = [], []

        for index, data, weight in zip(batch_data_indexes,
                                       batch_data_list,
                                       weights):

            if 'q_hat_input' in data:
                need_update.append(data)
                need_update_indexes.append(index)
                weights_update.append(weight)

            else:
                others.append(data)
                other_indexes.append(index)
                weights_other.append(weight)

        if need_update:

            q_hat_batch = prepare_input_for_batch(need_update,
                                                  device=self.model_device,
                                                  with_target=False)

            q_hat_output = self.target_network(**q_hat_batch['model_inputs']).detach()

            q_hat_values = t.gather(q_hat_output,
                                    dim=1,
                                    index=q_hat_batch['targets']['targets_idx'].unsqueeze(1)).cpu()


            for i, data in enumerate(need_update):
                data['target'] = q_hat_values[i].item() + data['reward']

        for data in others:
            data['target'] = data['reward']

        batch_data = need_update + others
        data_indexes = need_update_indexes + other_indexes
        weights = weights_update + weights_other

        weights = t.tensor(weights, device=self.model_device)

        batch = prepare_input_for_batch(batch_data, self.model_device)

        return batch, data_indexes, weights

    def sample_indexes(self):
        """
        Behaves differently from choices as the same index can't be sampled
        multiple time

        Returns:
            list: list of sampled indexes
        """
        buff_len = len(self.buffer)

        sampling_scores = self.sampling_scores[:buff_len]
        sampling_ranks_normalized =  (1 / (sampling_scores.argsort(descending=True).argsort() + 1)) ** self.alpha_sampling
        sampling_probas = sampling_ranks_normalized / sampling_ranks_normalized.sum()
        probas_cpu = sampling_probas.cpu().numpy()

        chosen_indexes = np.random.choice(buff_len,
                                          p=probas_cpu,
                                          size=self.batch_size,
                                          replace=False)

        sampled_probas = np.array([probas_cpu[i] for i in chosen_indexes])

        weights = (buff_len * sampled_probas) ** -self.beta_sampling
        weights = weights / weights.max()

        return chosen_indexes, weights


    def clean_action_data_buffer_and_sampling(self):
        """
        drops old data in buffer and sampling scores.
        Used because we now use a tensor for sampling scores
        """

        if len(self.buffer) + len(self.previous_actions_data) + 1 >= self.buffer_size:

            self.buffer = self.buffer[self.buffer_size // 2 :]
            sampling_scores = self.sampling_scores.cpu().numpy()

            sampling_scores = sampling_scores[self.buffer_size // 2 :]
            filling = np.array([nan] * (self.sampling_scores.size(0) - len(sampling_scores)))
            sampling_scores = np.hstack([sampling_scores, filling])

            assert sampling_scores.shape[0] == self.sampling_scores.size(0)

            self.sampling_scores = t.tensor(sampling_scores,
                                            dtype=t.float,
                                            device=self.model_device)