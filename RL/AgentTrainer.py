import torch, pdb, pickle, utils, sys 
#from collections import deque 
#import numpy as np 
from data.DataFeed import AtariDataFeed
from model.policy_model import PolicyModel 
from model.value_model import ValueModel 
from model.q_learning import QLearning 
from config.agent_config import AgentConfig

class AgentTrainer 
    """
    main class to train agents 
    """
    
    def __init(self, agent_config):
        self.agent_config = agent_config
        self.data_feed = self.agent_config.data_feed
        self.policy_model = self.agent_config.policy_model
        self.q_learning = self.agent_config.q_learning
        self.replay_buffer = self.agent_config.replay_buffer
        self.meta_learner_trainer = self.agent_config.meta_learner_trainer

        self.policy_optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=self.agent_config.policy_lr) 
        self.q_optimizer = torch.optim.Adam(self.q_learning.q_model.parameters(), lr = self.agent_config.q_lr)

    def train():
        init_action = self.initial_action() 
        init_state = self.initial_state() 
        state, reward, terminated, _, info = self.data_feed(initial_state, initial_action)
        while not terminated: 
            ## -- sample action from policy 
            action = self.policy_model(state) 
            
            #data-feed: abstract into the step function, given action and time-step, and current state, give the next state and the reward
            state, reward, terminated, _, info = self.data_feed(initial_state, initial_action)

            ## -- manage replay buffer 
            self.replay_buffer.insert(state, action, next_state) 
            
            prev_states_tensor, actions_tensor, states_tensor = self.replay_buffer.get_batch(B) 
            
            next_actions_tensor = policy_model.forward_tensor(policy_function, states_tensor)
            
            Q_future = self.q_learning.Q_future_function(states_tensor, next_actions_tensor) 
            y = reward + self.agent_config.gamma * Q_future 
            loss = torch.nn.mse_loss(y, Q(prev_states_tensor, actions_tensor)) 
            
            self.q_optimizer.zero_grad() 
            loss.backward()
            self.q_optimizer.step()
            
            if t % self.agent_config.policy_update_interval == 0: 
                prev_states_tensor, actions_tensor, states_tensor = self.replay_buffer.get_batch(B) 
                next_actions_sampled = self.policy_model_d.forward_tensor(policy_function, states_tensor) 

                q = self.q_learning.Q_function(states_tensor, next_actions_sampled)
                loss = -1.0 * q
                self.policy_optimizer.zero_grad()
                loss.backward()
                self.policy_optimizer.step()
        
            if t % meta_update_interval == 0:
                self.meta_learner_trainer.update_meta_learning_model(states_tensor, Q)

