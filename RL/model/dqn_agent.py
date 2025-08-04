import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import ale_py
import random
import os

from .dqn_network import DQN
from .data_buffer import ReplayBuffer, FrameStack

# Register Atari environments
gym.register_envs(ale_py)


class DQNAgent:
    """DQN Agent with delayed double Q-learning"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Environment setup
        self.env = gym.make(config.env_name)
        self.n_actions = self.env.action_space.n
        
        # Frame preprocessing
        self.frame_stack = FrameStack(config.frame_stack)
        
        # Networks - Double DQN setup
        obs_shape = (config.frame_stack * 3, 210, 160)  # RGB channels * stack size
        self.q_network = DQN(obs_shape, self.n_actions).to(self.device)
        self.target_network = DQN(obs_shape, self.n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Training setup
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        self.replay_buffer = ReplayBuffer(config.memory_size)
        self.epsilon = config.epsilon_start
        self.steps_done = 0
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax(1).item()
    
    def update_q_network(self):
        """Update Q-network using experience replay and double Q-learning"""
        if len(self.replay_buffer) < self.config.min_replay_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double Q-learning: use online network to select actions, target network to evaluate
        with torch.no_grad():
            # Select best actions using online network
            next_actions = self.q_network(next_states).argmax(1)
            # Evaluate those actions using target network
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + self.config.gamma * next_q_values * (1 - dones.unsqueeze(1))
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with current Q-network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train(self, num_episodes: int):
        """Train the DQN agent"""
        episode_rewards = []
        losses = []
        
        for episode in range(num_episodes):
            # Reset environment
            obs, _ = self.env.reset()
            state = self.frame_stack.reset(obs)
            episode_reward = 0
            episode_losses = []
            
            done = False
            while not done:
                # Select and perform action
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Stack frames
                next_state = self.frame_stack.append(next_obs)
                
                # Store transition
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                # Update state
                state = next_state
                episode_reward += reward
                self.steps_done += 1
                
                # Update Q-network at specified intervals
                if self.steps_done % self.config.policy_update_interval == 0:
                    loss = self.update_q_network()
                    if loss is not None:
                        episode_losses.append(loss)
                
                # Update target network at specified intervals (delayed update)
                if self.steps_done % self.config.target_update_freq == 0:
                    self.update_target_network()
                    print(f"Updated target network at step {self.steps_done}")
                
                # Save checkpoint
                if self.steps_done % self.config.save_interval == 0:
                    self.save_checkpoint(episode)
            
            # Update epsilon
            self.epsilon = max(self.config.epsilon_end, 
                             self.epsilon * self.config.epsilon_decay)
            
            # Record statistics
            episode_rewards.append(episode_reward)
            if episode_losses:
                losses.append(np.mean(episode_losses))
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}, Steps: {self.steps_done}")
        
        return episode_rewards, losses
    
    def save_checkpoint(self, episode: int):
        """Save model checkpoint"""
        checkpoint = {
            'episode': episode,
            'steps_done': self.steps_done,
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'config': self.config
        }
        path = os.path.join(self.config.checkpoint_dir, f'checkpoint_step_{self.steps_done}.pth')
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        print(f"Loaded checkpoint from {path}")
    
    def test(self, num_episodes: int = 5, render: bool = True):
        """Test the trained agent"""
        if render:
            test_env = gym.make(self.config.env_name, render_mode='human')
        else:
            test_env = self.env
        
        episode_rewards = []
        
        for episode in range(num_episodes):
            obs, _ = test_env.reset()
            state = self.frame_stack.reset(obs)
            episode_reward = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.q_network(state_tensor)
                    action = q_values.argmax(1).item()
                
                next_obs, reward, terminated, truncated, _ = test_env.step(action)
                done = terminated or truncated
                state = self.frame_stack.append(next_obs)
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
            print(f"Test Episode {episode + 1}: Reward = {episode_reward}")
        
        if render:
            test_env.close()
        
        return episode_rewards