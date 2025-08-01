# rainbow.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os

from networks import CategoricalDQN
from replay_buffer import PrioritizedReplayBuffer, NStepBuffer
from envs import make_atari_env
from utils import LinearSchedule, update_target_network, compute_categorical_loss, save_checkpoint, Logger
from config import RainbowConfig

class RainbowAgent:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # Environment
        self.env = make_atari_env(config.env_name, config.seed, config.frame_stack)
        self.num_actions = self.env.action_space.n
        
        # Networks - Categorical DQN with all Rainbow components
        self.model = CategoricalDQN(
            config.frame_stack, 
            self.num_actions,
            config.atom_size,
            config.v_min,
            config.v_max,
            dueling=config.dueling,
            noisy=config.noisy_nets,
            sigma_init=config.sigma_init
        ).to(self.device)
        
        self.target_model = CategoricalDQN(
            config.frame_stack, 
            self.num_actions,
            config.atom_size,
            config.v_min,
            config.v_max,
            dueling=config.dueling,
            noisy=config.noisy_nets,
            sigma_init=config.sigma_init
        ).to(self.device)
        
        update_target_network(self.model, self.target_model)
        
        # Move support to device
        self.model.z = self.model.z.to(self.device)
        self.target_model.z = self.target_model.z.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr, eps=config.eps)
        
        # Replay buffer
        if config.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                config.buffer_size,
                config.prioritized_replay_alpha,
                config.prioritized_replay_beta0,
                config.prioritized_replay_beta_steps
            )
        else:
            self.replay_buffer = NStepBuffer(
                config.buffer_size,
                config.n_step,
                config.gamma
            )
        
        # N-step buffer for multi-step returns
        self.n_step_buffer = deque(maxlen=config.n_step)
        
        # Logger
        self.logger = Logger(config.log_dir)
        
        # Training variables
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)
        
        # Beta schedule for prioritized replay
        if config.prioritized_replay:
            self.beta_schedule = LinearSchedule(
                config.prioritized_replay_beta0,
                1.0,
                config.prioritized_replay_beta_steps
            )
    
    def select_action(self, state):
        """Select action using the distributional Q-network"""
        # Handle LazyFrames
        if hasattr(state, '_force'):
            state = state._force()
        state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.config.noisy_nets:
                # Reset noise
                self.model.reset_noise()
            
            # Get Q-values from categorical distribution
            q_values = self.model.get_q_values(state_tensor)
            action = q_values.argmax(1).item()
        
        return action
    
    def compute_n_step_return(self, n_step_buffer):
        """Compute n-step return"""
        reward = sum([self.config.gamma ** i * t[2] for i, t in enumerate(n_step_buffer)])
        state, action = n_step_buffer[0][0], n_step_buffer[0][1]
        next_state, done = n_step_buffer[-1][3], n_step_buffer[-1][4]
        return state, action, reward, next_state, done
    
    def train_step(self):
        """Perform one training step"""
        if len(self.replay_buffer) < self.config.batch_size:
            return None
        
        # Sample batch
        if self.config.prioritized_replay:
            beta = self.beta_schedule.get_value(self.total_steps)
            states, actions, rewards, next_states, dones, indices, weights = \
                self.replay_buffer.sample(self.config.batch_size, beta)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.config.batch_size)
            weights = torch.ones(self.config.batch_size).to(self.device)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute categorical loss
        delta_z = (self.config.v_max - self.config.v_min) / (self.config.atom_size - 1)
        
        with torch.no_grad():
            # Reset noise for target network
            if self.config.noisy_nets:
                self.target_model.reset_noise()
            
            # Get next distribution
            next_dist = self.target_model(next_states)
            next_action = next_dist.mul(self.target_model.z).sum(2).argmax(1)
            next_dist = next_dist.gather(1, next_action.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.config.atom_size)).squeeze(1)
            
            # Compute projected distribution
            tz = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.config.gamma ** self.config.n_step * self.target_model.z.unsqueeze(0)
            tz = tz.clamp(self.config.v_min, self.config.v_max)
            b = (tz - self.config.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.config.atom_size - 1)) * (l == u)] += 1
            
            m = torch.zeros_like(next_dist)
            offset = torch.linspace(0, (m.size(0) - 1) * self.config.atom_size, m.size(0)).long().unsqueeze(1).to(self.device)
            
            m.view(-1).index_add_(0, (offset + l).view(-1), (next_dist * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (offset + u).view(-1), (next_dist * (b - l.float())).view(-1))
        
        # Get current distribution
        dist = self.model(states)
        log_dist = torch.log(dist + 1e-8)
        action_dist = log_dist.gather(1, actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.config.atom_size)).squeeze(1)
        
        # Compute loss
        loss = -(m * action_dist).sum(1)
        
        # Priority update
        if self.config.prioritized_replay:
            priorities = loss.detach().cpu().numpy() + 1e-6
            self.replay_buffer.update_priorities(indices, priorities)
        
        # Weighted loss
        loss = (loss * weights).mean()
        
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
            action = self.select_action(state)
            
            # Environment step
            next_state, reward, done, _ = self.env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # Store in n-step buffer
            self.n_step_buffer.append((state, action, reward, next_state, done))
            
            # Store transition when n-step buffer is full
            if len(self.n_step_buffer) == self.config.n_step:
                n_state, n_action, n_reward, n_next_state, n_done = self.compute_n_step_return(self.n_step_buffer)
                self.replay_buffer.push(n_state, n_action, n_reward, n_next_state, n_done)
            
            # Train
            if step > self.config.learning_starts and step % self.config.train_freq == 0:
                loss = self.train_step()
            
            # Update target network
            if step % self.config.target_update_freq == 0:
                update_target_network(self.model, self.target_model)
            
            # Reset if done
            if done:
                # Empty n-step buffer
                while len(self.n_step_buffer) > 0:
                    n_state, n_action, n_reward, n_next_state, n_done = self.compute_n_step_return(self.n_step_buffer)
                    self.replay_buffer.push(n_state, n_action, n_reward, n_next_state, n_done)
                    self.n_step_buffer.popleft()
                
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
                stats['buffer_size'] = len(self.replay_buffer)
                stats['episodes'] = episode_count
                if self.config.prioritized_replay:
                    stats['beta'] = self.beta_schedule.get_value(step)
                self.logger.log_metrics(step, stats)
            
            # Save checkpoint
            if step % self.config.save_interval == 0:
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'step': step,
                    'episode': episode_count,
                }
                save_checkpoint(checkpoint, os.path.join(self.config.save_dir, f'rainbow_{step}.pth'))
            
            self.total_steps = step
    
    def evaluate(self, num_episodes=10):
        """Evaluate the agent"""
        eval_rewards = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.select_action(state)
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
        
        return np.mean(eval_rewards), np.std(eval_rewards)

def main():
    config = RainbowConfig()
    
    # Create directories
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Set random seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Create and train agent
    agent = RainbowAgent(config)
    agent.train()
    
    # Evaluate final performance
    mean_reward, std_reward = agent.evaluate()
    print(f"Final evaluation: {mean_reward:.2f} +/- {std_reward:.2f}")

if __name__ == "__main__":
    main()