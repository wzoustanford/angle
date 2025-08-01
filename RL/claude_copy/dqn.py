# dqn.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os

from networks import DQN
from replay_buffer import ReplayBuffer
from envs import make_atari_env
from utils import LinearSchedule, update_target_network, compute_td_loss, compute_double_dqn_loss, save_checkpoint, Logger
from config import DQNConfig

class DQNAgent:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # Environment
        self.env = make_atari_env(config.env_name, config.seed, config.frame_stack)
        self.num_actions = self.env.action_space.n
        
        # Networks
        self.model = DQN(config.frame_stack, self.num_actions, 
                        dueling=False, noisy=False).to(self.device)
        self.target_model = DQN(config.frame_stack, self.num_actions,
                               dueling=False, noisy=False).to(self.device)
        update_target_network(self.model, self.target_model)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr, eps=config.eps)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        
        # Epsilon schedule
        self.epsilon_schedule = LinearSchedule(
            config.epsilon_start, 
            config.epsilon_end, 
            config.epsilon_decay
        )
        
        # Logger
        self.logger = Logger(config.log_dir)
        
        # Training variables
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)
        
    def select_action(self, state, epsilon):
        """Select action using epsilon-greedy policy"""
        if random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return q_values.argmax(1).item()
    
    def train_step(self):
        """Perform one training step"""
        if len(self.replay_buffer) < self.config.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute loss
        if self.config.double_dqn:
            loss = compute_double_dqn_loss(
                self.model, self.target_model, states, actions, 
                rewards, next_states, dones, self.config.gamma
            )
        else:
            loss = compute_td_loss(
                self.model, self.target_model, states, actions,
                rewards, next_states, dones, self.config.gamma
            )
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        
        return loss.item()
    
    def train(self):
        """Main training loop"""
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_count = 0
        
        for step in range(self.config.num_env_steps):
            # Select action
            epsilon = self.epsilon_schedule.get_value(step)
            action = self.select_action(state, epsilon)
            
            # Environment step
            next_state, reward, done, _ = self.env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # Store transition
            self.replay_buffer.push(state, action, reward, next_state, done)
            
            # Train
            if step > self.config.learning_starts and step % self.config.train_freq == 0:
                loss = self.train_step()
            
            # Update target network
            if step % self.config.target_update_freq == 0:
                update_target_network(self.model, self.target_model)
            
            # Reset if done
            if done:
                self.episode_rewards.append(episode_reward)
                self.logger.log_episode(episode_reward, episode_length)
                episode_count += 1
                episode_reward = 0
                episode_length = 0
                state = self.env.reset()
            else:
                state = next_state
            
            # Logging
            if step % self.config.log_interval == 0 and len(self.episode_rewards) > 0:
                stats = self.logger.get_stats()
                stats['epsilon'] = epsilon
                stats['buffer_size'] = len(self.replay_buffer)
                stats['episodes'] = episode_count
                self.logger.log_metrics(step, stats)
            
            # Save checkpoint
            if step % self.config.save_interval == 0:
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'step': step,
                    'episode': episode_count,
                }
                save_checkpoint(checkpoint, os.path.join(self.config.save_dir, f'dqn_{step}.pth'))
            
            self.total_steps = step
    
    def evaluate(self, num_episodes=10):
        """Evaluate the agent"""
        eval_rewards = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.select_action(state, epsilon=0.05)  # Small epsilon for evaluation
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
        
        return np.mean(eval_rewards), np.std(eval_rewards)

def main():
    config = DQNConfig()
    
    # Create directories
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Set random seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Create and train agent
    agent = DQNAgent(config)
    agent.train()
    
    # Evaluate final performance
    mean_reward, std_reward = agent.evaluate()
    print(f"Final evaluation: {mean_reward:.2f} +/- {std_reward:.2f}")

if __name__ == "__main__":
    main()