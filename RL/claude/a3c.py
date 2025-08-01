# a3c.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import os
from collections import deque

from networks import ActorCritic
from envs import make_atari_env
from utils import save_checkpoint, Logger, init_weights
from config import A3CConfig

class SharedAdam(optim.Adam):
    """Shared Adam optimizer for A3C"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1).share_memory_()
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()

def ensure_shared_grads(model, shared_model):
    """Ensure grads are shared between processes"""
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

class A3CWorker(mp.Process):
    """Worker process for A3C"""
    def __init__(self, rank, config, shared_model, optimizer, counter, lock):
        super(A3CWorker, self).__init__()
        self.rank = rank
        self.config = config
        self.shared_model = shared_model
        self.optimizer = optimizer
        self.counter = counter
        self.lock = lock
        
    def run(self):
        # Set seed
        torch.manual_seed(self.config.seed + self.rank)
        np.random.seed(self.config.seed + self.rank)
        
        # Create environment
        env = make_atari_env(
            self.config.env_name,
            self.config.seed + self.rank,
            self.config.frame_stack
        )
        
        # Create local model
        model = ActorCritic(
            self.config.frame_stack,
            env.action_space.n
        )
        model.train()
        
        # Episode variables
        state = env.reset()
        done = True
        episode_length = 0
        episode_reward = 0
        
        while self.counter.value < self.config.num_env_steps:
            # Sync with shared model
            model.load_state_dict(self.shared_model.state_dict())
            
            # Storage
            values = []
            log_probs = []
            rewards = []
            entropies = []
            
            # Rollout
            for step in range(self.config.num_steps):
                # Handle LazyFrames
                if hasattr(state, '_force'):
                    state_array = np.array(state._force())
                else:
                    state_array = state
                state_tensor = torch.FloatTensor(state_array).unsqueeze(0)
                
                # Forward pass
                logits, value = model(state_tensor)
                probs = F.softmax(logits, dim=-1)
                log_prob = F.log_softmax(logits, dim=-1)
                entropy = -(log_prob * probs).sum(1)
                
                # Sample action
                action = probs.multinomial(1).squeeze(1)
                
                # Environment step
                next_state, reward, done, _ = env.step(action.item())
                episode_reward += reward
                episode_length += 1
                
                # Store
                values.append(value)
                log_probs.append(log_prob[0, action])
                rewards.append(reward)
                entropies.append(entropy)
                
                # Update state
                state = next_state
                
                # Increment counter
                with self.lock:
                    self.counter.value += 1
                
                if done:
                    if self.rank == 0:
                        print(f"Process {self.rank}, Episode reward: {episode_reward}, Length: {episode_length}")
                    episode_reward = 0
                    episode_length = 0
                    state = env.reset()
                    break
            
            # Compute returns
            R = 0
            if not done:
                # Handle LazyFrames
                if hasattr(state, '_force'):
                    state_array = np.array(state._force())
                else:
                    state_array = state
                state_tensor = torch.FloatTensor(state_array).unsqueeze(0)
                _, value = model(state_tensor)
                R = value.item()
            
            # Accumulate gradients
            values.append(torch.tensor([R]))
            policy_loss = 0
            value_loss = 0
            gae = 0
            
            for i in reversed(range(len(rewards))):
                R = self.config.gamma * R + rewards[i]
                advantage = R - values[i].item()
                
                # Losses
                value_loss = value_loss + 0.5 * advantage ** 2
                policy_loss = policy_loss - log_probs[i] * advantage - self.config.entropy_coef * entropies[i]
            
            # Optimize
            self.optimizer.zero_grad()
            total_loss = policy_loss + self.config.value_loss_coef * value_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
            
            # Share gradients
            ensure_shared_grads(model, self.shared_model)
            self.optimizer.step()

class A3CMonitor(mp.Process):
    """Monitor process for logging and evaluation"""
    def __init__(self, config, shared_model, counter):
        super(A3CMonitor, self).__init__()
        self.config = config
        self.shared_model = shared_model
        self.counter = counter
        
    def run(self):
        logger = Logger(self.config.log_dir)
        env = make_atari_env(
            self.config.env_name,
            self.config.seed + 100,
            self.config.frame_stack
        )
        
        model = ActorCritic(
            self.config.frame_stack,
            env.action_space.n
        )
        
        while self.counter.value < self.config.num_env_steps:
            if self.counter.value % self.config.log_interval == 0:
                # Sync model
                model.load_state_dict(self.shared_model.state_dict())
                
                # Evaluate
                rewards = []
                for _ in range(5):
                    state = env.reset()
                    done = False
                    episode_reward = 0
                    
                    while not done:
                        # Handle LazyFrames
                        if hasattr(state, '_force'):
                            state_array = np.array(state._force())
                        else:
                            state_array = state
                        state_tensor = torch.FloatTensor(state_array).unsqueeze(0)
                        with torch.no_grad():
                            logits, _ = model(state_tensor)
                            probs = F.softmax(logits, dim=-1)
                            action = probs.argmax(1).item()
                        
                        state, reward, done, _ = env.step(action)
                        episode_reward += reward
                    
                    rewards.append(episode_reward)
                
                # Log
                stats = {
                    'mean_reward': np.mean(rewards),
                    'std_reward': np.std(rewards),
                    'steps': self.counter.value,
                }
                logger.log_metrics(self.counter.value, stats)
            
            # Save checkpoint
            if self.counter.value % self.config.save_interval == 0:
                checkpoint = {
                    'model_state_dict': self.shared_model.state_dict(),
                    'step': self.counter.value,
                }
                save_checkpoint(checkpoint, os.path.join(self.config.save_dir, f'a3c_{self.counter.value}.pth'))
            
                import time
            time.sleep(10)  # Check every 10 seconds

def train_a3c(config):
    """Main A3C training function"""
    # Create shared model
    env = make_atari_env(config.env_name, config.seed, config.frame_stack)
    shared_model = ActorCritic(
        config.frame_stack,
        env.action_space.n
    )
    shared_model.share_memory()
    shared_model.apply(init_weights)
    env.close()
    
    # Create shared optimizer
    optimizer = SharedAdam(shared_model.parameters(), lr=config.lr, eps=config.eps)
    
    # Create counter
    counter = mp.Value('i', 0)
    lock = mp.Lock()
    
    # Create processes
    processes = []
    
    # Start monitor
    monitor = A3CMonitor(config, shared_model, counter)
    monitor.start()
    processes.append(monitor)
    
    # Start workers
    for rank in range(config.num_processes):
        p = A3CWorker(rank, config, shared_model, optimizer, counter, lock)
        p.start()
        processes.append(p)
    
    # Wait for completion
    for p in processes:
        p.join()

def main():
    config = A3CConfig()
    
    # Create directories
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Set multiprocessing method
    mp.set_start_method('spawn')
    
    # Train
    train_a3c(config)

if __name__ == "__main__":
    main()