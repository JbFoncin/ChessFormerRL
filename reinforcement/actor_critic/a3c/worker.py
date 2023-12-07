"""the function to be parallelized and related tools
"""
from random import shuffle

from chess import Board

from modeling.tools.shared import move_data_to_device
from reinforcement.players import A2CModelPlayer
from reinforcement.reward import get_endgame_reward, get_move_reward


class HistoryRegisterer:
    
    def __init__(self, agent, competitor, board):
        """
        Args:
            agent (reinforcement.players.A2CModelPlayer): the agent
            competitor (reinforcement.players.PlayerABC): the opponent
            board (chess.Board): the current game state
        """        
        self.agent = agent
        self.competitor = competitor
        self.board = board
        self.episode_data = []
        
    def interact_and_register(self):
        
        """
        Each player chooses an action, reward is computed and data added to buffer

        Args:
            board (chess.Board): the current game

        Returns:
            chess.Board: the current game
            bool: True if game is finished
        """
        #agent plays
        player_output = self.agent.choose_action(self.board)

        reward = get_move_reward(self.board, player_output.action)

        self.board.push_san(player_output.action)

        # Chech if competitor can play and get reward

        endgame_reward, _ = get_endgame_reward(self.board, self.competitor.color)

        if endgame_reward is not None: #finish game

            reward += endgame_reward

            self.update_episode_data(player_output.inference_data,
                                     player_output.action_index,
                                     player_output.policy_score,
                                     player_output.estimated_state_value,
                                     reward)

            return reward, False

        competitor_output = self.competitor.choose_action(self.board)

        reward -= get_move_reward(self.board, competitor_output.action)

        self.board.push_san(competitor_output.action)

        # check if the game is finished after competitor's action

        endgame_reward, neutral = get_endgame_reward(self.board, self.agent.color)

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

            return reward, False

        self.update_episode_data(player_output.inference_data,
                                 player_output.action_index,
                                 player_output.policy_score,
                                 player_output.estimated_state_value,
                                 reward)

        return reward, True
    

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

        self.episode_data.append({**model_inputs,
                                  'reward': current_reward,
                                  'action_index': current_action_index,
                                  'action_policy_score': action_policy_score,
                                  'estimated_state_value': estimated_state_value})
        
        
    def compute_state_value_target(self):
        """
        computes state value target with history data
        """        
        for step, next_step in zip(self.episode_data[:-1],
                                   self.episode_data[1:]):
        
            step['state_value_target'] = step['reward'] + next_step['estimated_state_value']
            
        
        self.episode_data[-1]['state_value_target'] = self.episode_data[-1]['reward']
        
        
        
    
def gather_data(model, competitor, model_device, output_queue, board_fen=None):
    """
    generates training data by exploring environment

    Args:
        model (torch.nn.Module): the Actor Critic neural network
        competitor (renforcement.player): the opponent
        model_device (str): 'cuda' or 'cpu', the device where is the model
        board_fen (str or None): string of encoded board as string
        
    Returns:
        list[dict]: one dict for each action in the game
    """
    
    agent = A2CModelPlayer(model, model_device)
    
    colors = [True, False]
    shuffle(colors)
    color_agent, color_competitor = colors
    # True for white and False for black like in the chess package
    agent.set_color(color_agent)
    competitor.set_color(color_competitor)
    
    if board_fen:
    
        board = Board(board_fen)
        
    else:
        
        board = Board()

    if competitor.color: #if competitor plays first

        competitor_output = competitor.choose_action(board)
        board.push_san(competitor_output.action)
        
    episode_registerer = HistoryRegisterer(agent, competitor, board)
    
    episode_reward = 0
    
    game_continues = True
    
    while game_continues:
        
        reward, game_continues = episode_registerer.interact_and_register()
        
        episode_reward += reward
        
    episode_registerer.compute_state_value_target()
                                
    output_queue.put({'episode_data': episode_registerer.episode_data,
                      'episode_reward': episode_reward})