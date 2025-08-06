import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import os
from typing import Dict, List, Optional

from .dqn_network import DQN
from .dueling_dqn_network import DuelingDQN
from .distributed_buffer import DistributedReplayBuffer
from .parallel_env_manager import ParallelEnvironmentManager


class DistributedDQNAgent:
    """Distributed DQN Agent that trains from multiple parallel environments"""
    
    def __init__(self, config, num_workers: int = 4, use_dueling: bool = False):
        self.config = config
        self.num_workers = num_workers
        self.use_dueling = use_dueling
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Setup networks
        obs_shape = (config.frame_stack * 3, 210, 160)  # RGB channels * stack size
        n_actions = self._get_n_actions()
        
        if use_dueling:
            print("Using Dueling DQN Network")
            self.q_network = DuelingDQN(obs_shape, n_actions).to(self.device)
            self.target_network = DuelingDQN(obs_shape, n_actions).to(self.device)
        else:
            print("Using Standard DQN Network")
            self.q_network = DQN(obs_shape, n_actions).to(self.device)
            self.target_network = DQN(obs_shape, n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Training setup
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
        # Distributed components
        self.replay_buffer = DistributedReplayBuffer(config.memory_size)
        self.env_manager = ParallelEnvironmentManager(
            config=config,
            replay_buffer=self.replay_buffer,
            num_workers=num_workers,
            shared_network=self.q_network
        )
        
        # Training state
        self.steps_done = 0
        self.episodes_done = 0
        self.epsilon = config.epsilon_start
        
        # Statistics
        self.training_stats = {
            'losses': [],
            'rewards_per_worker': [[] for _ in range(num_workers)],
            'collection_times': [],
            'training_times': [],
            'buffer_fill_ratios': []
        }
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    def _get_n_actions(self):
        """Get number of actions from environment"""
        import gymnasium as gym
        import ale_py
        gym.register_envs(ale_py)
        temp_env = gym.make(self.config.env_name)
        n_actions = temp_env.action_space.n
        temp_env.close()
        return n_actions
    
    def get_epsilon_schedule(self):
        """Create epsilon schedule function"""
        def epsilon_fn(episode):
            # Decay epsilon based on total episodes across all workers
            decay_rate = self.config.epsilon_decay
            epsilon = max(
                self.config.epsilon_end,
                self.config.epsilon_start * (decay_rate ** episode)
            )
            return epsilon
        return epsilon_fn
    
    def update_q_network(self, prioritize_recent: bool = True):
        """Update Q-network using experience replay"""
        if len(self.replay_buffer) < self.config.min_replay_size:
            return None
        
        # Sample batch from distributed buffer
        batch_data = self.replay_buffer.sample(
            self.config.batch_size, 
            prioritize_recent=prioritize_recent
        )
        
        if batch_data is None:
            return None
        
        states, actions, rewards, next_states, dones, worker_ids = batch_data
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double Q-learning
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + self.config.gamma * next_q_values * (1 - dones.unsqueeze(1))
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        # Also update the shared network for workers
        self.env_manager.update_shared_network(self.q_network)
    
    def train_distributed(self, total_episodes: int, collection_interval: int = 50, max_time_seconds: float = None):
        """Train using distributed experience collection"""
        print(f"Starting distributed training with {self.num_workers} workers")
        print(f"Target episodes: {total_episodes}")
        
        # Start continuous collection
        self.env_manager.start_collection(
            episodes_per_worker=None,  # Continuous
            epsilon_schedule=self.get_epsilon_schedule()
        )
        
        training_start_time = time.time()
        last_stats_time = time.time()
        
        try:
            while (self.episodes_done < total_episodes and 
                   (max_time_seconds is None or time.time() - training_start_time < max_time_seconds)):
                # Let workers collect experiences
                time.sleep(1)  # Brief pause
                
                # Train on collected experiences
                training_time_start = time.time()
                
                # Perform multiple training updates
                for _ in range(self.config.policy_update_interval):
                    if len(self.replay_buffer) >= self.config.min_replay_size:
                        loss = self.update_q_network()
                        if loss is not None:
                            self.training_stats['losses'].append(loss)
                        self.steps_done += 1
                
                training_time = time.time() - training_time_start
                self.training_stats['training_times'].append(training_time)
                
                # Update target network periodically
                if self.steps_done % self.config.target_update_freq == 0:
                    self.update_target_network()
                    print(f"Updated target network at step {self.steps_done}")
                
                # Save checkpoint periodically
                if self.steps_done % self.config.save_interval == 0:
                    self.save_checkpoint()
                
                # Print statistics periodically - make it more frequent for short experiments
                if time.time() - last_stats_time > 5:  # Every 5 seconds for better tracking
                    stats = self.env_manager.get_statistics()
                    print(f"Distributed Training Progress:")
                    print(f"  Total Episodes: {stats['total_episodes']}")
                    print(f"  Total Steps: {stats['total_steps']}")
                    print(f"  Per-Worker Episodes: {[w['total_episodes'] for w in stats['worker_stats']]}")
                    print(f"  Per-Worker Steps: {[w['total_steps'] for w in stats['worker_stats']]}")
                    print(f"  Average Reward: {stats['overall_recent_avg']:.2f}")
                    print(f"  Buffer Size: {len(self.replay_buffer)}")
                    last_stats_time = time.time()
                
                # Update episode count from workers
                stats = self.env_manager.get_statistics()
                self.episodes_done = stats['total_episodes']
                
                # Record buffer fill ratio
                buffer_stats = self.replay_buffer.get_statistics()
                self.training_stats['buffer_fill_ratios'].append(buffer_stats['fill_ratio'])
        
        finally:
            # Stop collection
            self.env_manager.stop_collection()
            
        total_training_time = time.time() - training_start_time
        
        # Print final detailed statistics
        final_stats = self.env_manager.get_statistics()
        print(f"\nDistributed Training Completed in {total_training_time:.2f} seconds")
        print(f"Final Statistics:")
        print(f"  Total Episodes: {final_stats['total_episodes']}")
        print(f"  Total Steps: {final_stats['total_steps']}")
        print(f"  Steps per Episode: {final_stats['total_steps'] / max(1, final_stats['total_episodes']):.1f}")
        print(f"  Per-Worker Episodes: {[w['total_episodes'] for w in final_stats['worker_stats']]}")
        print(f"  Per-Worker Steps: {[w['total_steps'] for w in final_stats['worker_stats']]}")
        print(f"  Average Reward: {final_stats['overall_avg_reward']:.2f}")
        print(f"  Recent Average Reward: {final_stats['overall_recent_avg']:.2f}")
        
        return self.get_final_statistics()
    
    def train_batch_collection(self, total_episodes: int, episodes_per_batch: int = 20):
        """Train using batch-based collection (alternative approach)"""
        print(f"Starting batch-based distributed training")
        print(f"Episodes per batch: {episodes_per_batch}, Total episodes: {total_episodes}")
        
        batch_num = 0
        
        while self.episodes_done < total_episodes:
            batch_start_time = time.time()
            
            # Collect batch of experiences
            epsilon = max(
                self.config.epsilon_end,
                self.config.epsilon_start * (self.config.epsilon_decay ** self.episodes_done)
            )
            
            episode_results = self.env_manager.collect_batch(
                num_episodes_per_worker=episodes_per_batch // self.num_workers,
                epsilon=epsilon
            )
            
            collection_time = time.time() - batch_start_time
            self.training_stats['collection_times'].append(collection_time)
            
            # Update episode count
            self.episodes_done += len(episode_results)
            
            # Train on collected experiences
            training_start_time = time.time()
            
            for _ in range(episodes_per_batch * 2):  # Multiple training steps per batch
                if len(self.replay_buffer) >= self.config.min_replay_size:
                    loss = self.update_q_network()
                    if loss is not None:
                        self.training_stats['losses'].append(loss)
                    self.steps_done += 1
            
            training_time = time.time() - training_start_time
            self.training_stats['training_times'].append(training_time)
            
            # Update target network
            if batch_num % 5 == 0:  # Every 5 batches
                self.update_target_network()
            
            # Print batch statistics
            if batch_num % 10 == 0:
                self._print_batch_stats(batch_num, episode_results, collection_time, training_time)
            
            batch_num += 1
        
        return self.get_final_statistics()
    
    def _print_training_stats(self):
        """Print current training statistics"""
        stats = self.env_manager.get_statistics()
        buffer_stats = self.replay_buffer.get_statistics()
        
        recent_losses = self.training_stats['losses'][-100:] if self.training_stats['losses'] else [0]
        avg_loss = np.mean(recent_losses)
        
        print(f"\n=== Training Statistics ===")
        print(f"Episodes: {stats['total_episodes']}")
        print(f"Steps: {stats['total_steps']}")
        print(f"Training Steps: {self.steps_done}")
        print(f"Average Reward: {stats['overall_recent_avg']:.2f}")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Buffer Fill: {buffer_stats['fill_ratio']:.1%} ({buffer_stats['size']}/{buffer_stats['capacity']})")
        print(f"Worker Episodes: {[w['total_episodes'] for w in stats['worker_stats']]}")
        print("=" * 30)
    
    def _print_batch_stats(self, batch_num: int, episode_results: List, 
                          collection_time: float, training_time: float):
        """Print batch-specific statistics"""
        avg_reward = np.mean([r['episode_reward'] for r in episode_results])
        total_steps = sum([r['episode_length'] for r in episode_results])
        
        print(f"Batch {batch_num}: "
              f"Avg Reward: {avg_reward:.2f}, "
              f"Steps: {total_steps}, "
              f"Collection: {collection_time:.2f}s, "
              f"Training: {training_time:.2f}s")
    
    def save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint = {
            'episodes_done': self.episodes_done,
            'steps_done': self.steps_done,
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats,
            'buffer_stats': self.replay_buffer.get_statistics()
        }
        
        path = os.path.join(
            self.config.checkpoint_dir, 
            f'distributed_checkpoint_episodes_{self.episodes_done}.pth'
        )
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episodes_done = checkpoint['episodes_done']
        self.steps_done = checkpoint['steps_done']
        
        if 'training_stats' in checkpoint:
            self.training_stats = checkpoint['training_stats']
        
        # Update shared network for workers
        self.env_manager.update_shared_network(self.q_network)
        
        print(f"Loaded checkpoint from {path}")
    
    def get_final_statistics(self):
        """Get comprehensive final statistics"""
        env_stats = self.env_manager.get_statistics()
        buffer_stats = self.replay_buffer.get_statistics()
        
        return {
            'env_stats': env_stats,
            'buffer_stats': buffer_stats,
            'training_stats': self.training_stats,
            'final_episodes': self.episodes_done,
            'final_steps': self.steps_done,
            'average_collection_time': np.mean(self.training_stats['collection_times']) if self.training_stats['collection_times'] else 0,
            'average_training_time': np.mean(self.training_stats['training_times']) if self.training_stats['training_times'] else 0,
            'final_loss': np.mean(self.training_stats['losses'][-100:]) if self.training_stats['losses'] else 0
        }
    
    def test_distributed(self, num_episodes: int = 10):
        """Test the agent using parallel environments"""
        print(f"Testing agent with {num_episodes} episodes across {self.num_workers} workers")
        
        # Collect test episodes with no exploration
        test_results = self.env_manager.collect_batch(
            num_episodes_per_worker=num_episodes // self.num_workers,
            epsilon=0.0  # No exploration
        )
        
        rewards = [result['episode_reward'] for result in test_results]
        lengths = [result['episode_length'] for result in test_results]
        
        print(f"Test Results:")
        print(f"  Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"  Max Reward: {np.max(rewards):.2f}")
        print(f"  Min Reward: {np.min(rewards):.2f}")
        print(f"  Average Length: {np.mean(lengths):.2f}")
        
        return {
            'rewards': rewards,
            'lengths': lengths,
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards)
        }