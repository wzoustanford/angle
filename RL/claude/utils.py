# utils.py
import torch
import torch.nn as nn
import numpy as np
import os
import glob
from collections import deque

class LinearSchedule:
    """Linear interpolation between initial_value and final_value"""
    def __init__(self, initial_value, final_value, num_steps):
        self.initial_value = initial_value
        self.final_value = final_value
        self.num_steps = num_steps
    
    def get_value(self, step):
        fraction = min(float(step) / self.num_steps, 1.0)
        return self.initial_value + fraction * (self.final_value - self.initial_value)

def init_weights(m):
    """Initialize network weights"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0.0)

def update_target_network(current_model, target_model):
    """Copy weights from current model to target model"""
    target_model.load_state_dict(current_model.state_dict())

def compute_td_loss(model, target_model, states, actions, rewards, next_states, dones, gamma=0.99):
    """Compute TD loss for DQN"""
    current_q_values = model(states).gather(1, actions.unsqueeze(1))
    
    with torch.no_grad():
        next_q_values = target_model(next_states).max(1)[0]
        target_q_values = rewards + gamma * next_q_values * (1 - dones)
    
    loss = nn.functional.mse_loss(current_q_values.squeeze(), target_q_values)
    return loss

def compute_double_dqn_loss(model, target_model, states, actions, rewards, next_states, dones, gamma=0.99):
    """Compute TD loss for Double DQN"""
    current_q_values = model(states).gather(1, actions.unsqueeze(1))
    
    with torch.no_grad():
        next_actions = model(next_states).argmax(1)
        next_q_values = target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        target_q_values = rewards + gamma * next_q_values * (1 - dones)
    
    loss = nn.functional.mse_loss(current_q_values.squeeze(), target_q_values)
    return loss

def compute_categorical_loss(model, target_model, states, actions, rewards, next_states, dones, 
                           gamma=0.99, v_min=-10, v_max=10, atom_size=51):
    """Compute loss for Categorical DQN"""
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)
    
    delta_z = (v_max - v_min) / (atom_size - 1)
    z = torch.linspace(v_min, v_max, atom_size).to(states.device)
    
    # Get current distribution
    dist = model(states)
    action_dist = dist.gather(1, actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, atom_size)).squeeze(1)
    
    # Get next distribution
    with torch.no_grad():
        next_dist = target_model(next_states)
        next_action = next_dist.mul(z).sum(2).argmax(1)
        next_dist = next_dist.gather(1, next_action.unsqueeze(1).unsqueeze(2).expand(-1, -1, atom_size)).squeeze(1)
        
        # Compute projected distribution
        tz = rewards.unsqueeze(1) + gamma * (1 - dones.unsqueeze(1)) * z.unsqueeze(0)
        tz = tz.clamp(v_min, v_max)
        b = (tz - v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()
        
        l[(u > 0) * (l == u)] -= 1
        u[(l < (atom_size - 1)) * (l == u)] += 1
        
        m = torch.zeros_like(next_dist)
        offset = torch.linspace(0, (m.size(0) - 1) * atom_size, m.size(0)).long().unsqueeze(1).to(actions.device)
        
        m.view(-1).index_add_(0, (offset + l).view(-1), (next_dist * (u.float() - b)).view(-1))
        m.view(-1).index_add_(0, (offset + u).view(-1), (next_dist * (b - l.float())).view(-1))
    
    loss = -(m * action_dist.log()).sum(1).mean()
    return loss

def save_checkpoint(state, filename):
    """Save model checkpoint"""
    torch.save(state, filename)

def load_checkpoint(filename, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('episode', 0), checkpoint.get('step', 0)

def cleanup_log_dir(log_dir):
    """Create or clean up log directory"""
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)

class Logger:
    """Simple logger for tracking training metrics"""
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
    def log_episode(self, reward, length):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
    
    def log_metrics(self, step, metrics):
        """Log metrics to console and file"""
        msg = f"Step: {step}"
        for key, value in metrics.items():
            msg += f", {key}: {value:.4f}"
        print(msg)
        
        # Write to file
        with open(os.path.join(self.log_dir, "progress.txt"), "a") as f:
            f.write(msg + "\n")
    
    def get_stats(self):
        """Get current statistics"""
        if len(self.episode_rewards) > 0:
            return {
                "mean_reward": np.mean(self.episode_rewards),
                "mean_length": np.mean(self.episode_lengths),
                "max_reward": np.max(self.episode_rewards),
                "min_reward": np.min(self.episode_rewards),
            }
        else:
            return {
                "mean_reward": 0,
                "mean_length": 0,
                "max_reward": 0,
                "min_reward": 0,
            }