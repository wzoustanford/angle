import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import ale_py
import random
import os

from .dqn_network import DQN
from .data_buffer import ReplayBuffer, PrioritizedReplayBuffer, FrameStack
from .r2d2_network import R2D2Network
from .sequence_buffer import SequenceReplayBuffer, SequenceDataLoader
from .device_utils import get_device_manager

# Register Atari environments
gym.register_envs(ale_py)


class DQNAgent:
    """
    DQN Agent with support for both standard DQN and R2D2 modes
    
    Modes:
    - Standard DQN: use_r2d2=False (default)
    - R2D2: use_r2d2=True (LSTM + sequence replay)
    """
    def __init__(self, config):
        self.config = config
        self.devmgr = get_device_manager(getattr(config, 'device', None))
        self.device = self.devmgr.device
        
        # Environment setup
        self.env = gym.make(config.env_name)
        self.n_actions = self.env.action_space.n
        
        # Frame preprocessing
        self.frame_stack = FrameStack(config.frame_stack)
        
        # Networks - Support both DQN and R2D2
        obs_shape = (config.frame_stack * 3, 210, 160)  # RGB channels * stack size
        
        if getattr(config, 'use_r2d2', False):
            # R2D2 Network with LSTM
            self.q_network = self.devmgr.to_dev(R2D2Network(
                obs_shape, 
                self.n_actions,
                lstm_size=getattr(config, 'lstm_size', 512),
                num_lstm_layers=getattr(config, 'num_lstm_layers', 1)
            ))
            
            self.target_network = self.devmgr.to_dev(R2D2Network(
                obs_shape,
                self.n_actions,
                lstm_size=getattr(config, 'lstm_size', 512),
                num_lstm_layers=getattr(config, 'num_lstm_layers', 1)
            ))
            
            print(f"Using R2D2 Network (LSTM size: {getattr(config, 'lstm_size', 512)})")
        else:
            # Standard DQN
            self.q_network = self.devmgr.to_dev(DQN(obs_shape, self.n_actions))
            self.target_network = self.devmgr.to_dev(DQN(obs_shape, self.n_actions))
            print("Using Standard DQN Network")
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Training setup
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
        # Choose replay buffer type based on mode
        if getattr(config, 'use_r2d2', False):
            # R2D2: Use sequence-based replay buffer
            self.replay_buffer = SequenceReplayBuffer(
                capacity=config.memory_size,
                sequence_length=getattr(config, 'sequence_length', 80),
                burn_in_length=getattr(config, 'burn_in_length', 40),
                alpha=config.priority_alpha if config.use_prioritized_replay else 0.0,
                beta=config.priority_beta_start if config.use_prioritized_replay else 1.0,
                epsilon=config.priority_epsilon,
                priority_type=config.priority_type if config.use_prioritized_replay else 'uniform'
            )
            print(f"Using R2D2 Sequence Buffer (seq_len: {getattr(config, 'sequence_length', 80)})")
        else:
            # Standard DQN: Use transition-based replay buffer
            if config.use_prioritized_replay:
                self.replay_buffer = PrioritizedReplayBuffer(
                    capacity=config.memory_size,
                    alpha=config.priority_alpha,
                    beta=config.priority_beta_start,
                    epsilon=config.priority_epsilon,
                    priority_type=config.priority_type
                )
            else:
                self.replay_buffer = ReplayBuffer(config.memory_size)
            print(f"Using {'Prioritized' if config.use_prioritized_replay else 'Uniform'} Transition Buffer")
        
        if config.use_prioritized_replay:
            self.priority_beta = config.priority_beta_start
            self.priority_beta_end = config.priority_beta_end
        
        self.epsilon = config.epsilon_start
        self.steps_done = 0
        
        # R2D2 specific state
        self.hidden_state = None  # LSTM hidden state for current episode
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection with R2D2 support"""
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = self.devmgr.to_dev(torch.FloatTensor(state).unsqueeze(0))
                
                if getattr(self.config, 'use_r2d2', False):
                    # R2D2: Use LSTM for action selection
                    q_values, self.hidden_state = self.q_network.forward_single_step(
                        state_tensor, self.hidden_state
                    )
                else:
                    # Standard DQN
                    q_values = self.q_network(state_tensor)
                
                return q_values.argmax(1).item()
    
    def reset_hidden_state(self):
        """Reset LSTM hidden state at episode start (R2D2 only)"""
        if getattr(self.config, 'use_r2d2', False):
            self.hidden_state = None
    
    def update_q_network(self):
        """Update Q-network using experience replay (supports both DQN and R2D2)"""
        if len(self.replay_buffer) < self.config.min_replay_size:
            return None
        
        if getattr(self.config, 'use_r2d2', False):
            return self._update_r2d2()
        else:
            return self._update_dqn()
    
    def _update_r2d2(self):
        """R2D2 sequence-based update"""
        # Sample sequences
        if self.config.use_prioritized_replay:
            sequences, weights, idxs = self.replay_buffer.sample(self.config.batch_size)
            weights = self.devmgr.to_dev(torch.FloatTensor(weights))
        else:
            sequences, _, idxs = self.replay_buffer.sample(self.config.batch_size)
            weights = None
        
        # Prepare training data
        burn_in_data, target_data = SequenceDataLoader.prepare_training_batch(sequences)
        
        if target_data is None:
            return None
        
        # Convert to tensors
        burn_in_states = self.devmgr.to_dev(torch.FloatTensor(burn_in_data['states']))
        target_states = self.devmgr.to_dev(torch.FloatTensor(target_data['states']))
        target_actions = self.devmgr.to_dev(torch.LongTensor(target_data['actions']))
        target_rewards = self.devmgr.to_dev(torch.FloatTensor(target_data['rewards']))
        target_next_states = self.devmgr.to_dev(torch.FloatTensor(target_data['next_states']))
        target_dones = self.devmgr.to_dev(torch.FloatTensor(target_data['dones']))
        
        batch_size, seq_len = target_states.shape[:2]
        
        # Burn-in phase: warm up LSTM with burn-in data
        with torch.no_grad():
            if burn_in_data['states'].shape[1] > 0:  # If we have burn-in data
                _, hidden_state = self.q_network.forward(burn_in_states)
            else:
                hidden_state = self.q_network.init_hidden(batch_size, self.device)
        
        # Training phase: forward through target sequence
        current_q_values, _ = self.q_network.forward(target_states, hidden_state)
        
        # Get Q-values for actions taken: (batch, seq, 1)
        current_q_values = current_q_values.gather(-1, target_actions.unsqueeze(-1))
        
        # Target Q-values using target network
        with torch.no_grad():
            # Burn-in for target network
            if burn_in_data['states'].shape[1] > 0:
                _, target_hidden = self.target_network.forward(burn_in_states)
            else:
                target_hidden = self.target_network.init_hidden(batch_size, self.device)
            
            # Double Q-learning: select actions with online network
            next_q_online, _ = self.q_network.forward(target_next_states, hidden_state)
            next_actions = next_q_online.argmax(-1, keepdim=True)
            
            # Evaluate actions with target network
            next_q_target, _ = self.target_network.forward(target_next_states, target_hidden)
            next_q_values = next_q_target.gather(-1, next_actions)
            
            # Calculate targets
            target_q_values = target_rewards.unsqueeze(-1) + \
                            self.config.gamma * next_q_values * (1 - target_dones.unsqueeze(-1))
        
        # Calculate loss
        td_errors = target_q_values - current_q_values
        
        if weights is not None:
            # Prioritized replay: weight the loss
            loss = (weights.unsqueeze(-1).unsqueeze(-1) * (td_errors ** 2)).mean()
        else:
            loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()
        
        # Update priorities
        if self.config.use_prioritized_replay and idxs is not None:
            # Use mean TD error over sequence for priority
            priorities = td_errors.detach().cpu().numpy().mean(axis=(1, 2))
            self.replay_buffer.update_priorities(idxs, priorities)
        
        return loss.item()
    
    def _update_dqn(self):
        """Standard DQN update"""
        # Sample batch - handle both buffer types
        if self.config.use_prioritized_replay:
            states, actions, rewards, next_states, dones, weights, idxs = self.replay_buffer.sample(self.config.batch_size)
            weights = self.devmgr.to_dev(torch.FloatTensor(weights))
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)
            weights = None
            idxs = None
        
        # Convert to tensors
        states = self.devmgr.to_dev(torch.FloatTensor(states))
        actions = self.devmgr.to_dev(torch.LongTensor(actions))
        rewards = self.devmgr.to_dev(torch.FloatTensor(rewards))
        next_states = self.devmgr.to_dev(torch.FloatTensor(next_states))
        dones = self.devmgr.to_dev(torch.FloatTensor(dones))
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double Q-learning: use online network to select actions, target network to evaluate
        with torch.no_grad():
            # Select best actions using online network
            next_actions = self.q_network(next_states).argmax(1)
            # Evaluate those actions using target network
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + self.config.gamma * next_q_values * (1 - dones.unsqueeze(1))
        
        # Calculate TD errors for priority updates
        td_errors = target_q_values - current_q_values
        
        # Compute loss
        if weights is not None:
            # Weighted loss for prioritized replay
            loss = (weights.unsqueeze(1) * F.mse_loss(current_q_values, target_q_values, reduction='none')).mean()
        else:
            loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()
        
        # Update priorities for prioritized replay
        if self.config.use_prioritized_replay and idxs is not None:
            priorities = td_errors.detach().cpu().numpy().flatten()
            self.replay_buffer.update_priorities(idxs, priorities)
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with current Q-network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train(self, num_episodes: int):
        """Train the DQN agent"""
        episode_rewards = []
        losses = []
        
        for episode in range(num_episodes):
            # Reset environment and hidden state
            obs, _ = self.env.reset()
            state = self.frame_stack.reset(obs)
            self.reset_hidden_state()  # Reset LSTM state for R2D2
            episode_reward = 0
            episode_losses = []
            
            done = False
            while not done:
                # Select and perform action
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Clip rewards if configured (R2D2 option)
                if getattr(self.config, 'clip_rewards', False):
                    reward = np.clip(reward, -1.0, 1.0)
                
                # Stack frames
                next_state = self.frame_stack.append(next_obs)
                
                # Store transition (different for R2D2 vs DQN)
                if getattr(self.config, 'use_r2d2', False):
                    # R2D2: Add to sequence buffer
                    self.replay_buffer.push_transition(state, action, reward, next_state, done)
                else:
                    # DQN: Add to transition buffer
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
            
            # Update beta for prioritized replay
            if self.config.use_prioritized_replay:
                # Linear annealing of beta
                progress = min(episode / 100.0, 1.0)  # Anneal over 100 episodes
                self.priority_beta = self.config.priority_beta_start + progress * (
                    self.priority_beta_end - self.config.priority_beta_start)
                self.replay_buffer.update_beta(self.priority_beta)
            
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
                    state_tensor = self.devmgr.to_dev(torch.FloatTensor(state).unsqueeze(0))
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