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
        super().__init__(model_1, *args)
        self.loss = QRLoss(self.agent.model.nb_quantiles, kappa=kappa)
        self.loss = self.loss.to(self.model_device)
        
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
            # q_hat.gather(dim=1, index=t.tensor([[[x]*nb_quantiles], [[y]*nb_quantiles]]))
            # gives expected output for batch size 2 and best actions x and y
            idx = t.tensor(current_action_index).repeat(self.agent.model.nb_quantiles).unsqueeze(0).unsqueeze(1)

            q_hat_action = t.gather(q_hat, dim=1, index=idx)

            sampling_score = self.loss(to_buffer['estimated_action_value'] - to_buffer['reward'] - q_hat_action)
            self.sampling_scores[len(self.buffer) - 1] = sampling_score

        for element in self.previous_actions_data:
            element['reward'] += current_reward

        self.previous_actions_data.append({**model_inputs,
                                           'reward': current_reward,
                                           'target_idx': current_action_index,
                                           'estimated_action_value': estimated_action_value})    
    
    
    def train_batch(self):
        """
        samples and train one batch

        Returns:
            float: loss value on the current batch
        """
        self.optimizer.zero_grad()

        batch, sample_indexes, weights = self.make_training_batch()

        model_output = self.model(**batch['model_inputs'])
        
        gather_index = t.repeat_interleave(batch['targets']['targets_idx'],
                                            self.agent.model.nb_quantiles)
        
        gather_index_reshaped = gather_index.view(self.batch_size, -1).unsqueeze(1)

        predicted = t.gather(model_output, dim=1,
                                index=gather_index_reshaped)

        loss_per_batch = self.loss(predicted, batch['targets']['targets'], weights)

        loss_value = loss_per_batch.mean(0)
        loss_value.backward()

        new_sampling_scores = loss_per_batch.detach().cpu()

        self.optimizer.step()

        for index, value in zip(sample_indexes, new_sampling_scores):
            self.sampling_scores[index] = value.item()

        model_state_dict = self.model.state_dict()
        target_network_state_dict = self.target_network.state_dict()

        for key in target_network_state_dict:
            target_network_state_dict[key] = self.tau * model_state_dict[key] + (1 - self.tau) * target_network_state_dict[key]

        self.target_network.load_state_dict(target_network_state_dict)

        return loss_value.cpu().detach().item()
        
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