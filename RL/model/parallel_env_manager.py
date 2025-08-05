import gymnasium as gym
import ale_py
import numpy as np
import threading
import time
import queue
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from .distributed_buffer import DistributedReplayBuffer, DistributedFrameStack

# Register Atari environments
gym.register_envs(ale_py)


class EnvironmentWorker:
    """Worker that runs a single environment and collects experiences"""
    
    def __init__(self, worker_id: int, env_name: str, frame_stack: int, 
                 replay_buffer: DistributedReplayBuffer, shared_network=None):
        self.worker_id = worker_id
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.frame_stack = DistributedFrameStack(frame_stack)
        self.replay_buffer = replay_buffer
        self.shared_network = shared_network
        
        self.n_actions = self.env.action_space.n
        self.episode_count = 0
        self.total_steps = 0
        self.episode_rewards = []
        self.running = False
        
    def select_action(self, state, epsilon: float = 0.1):
        """Select action using epsilon-greedy or shared network"""
        if self.shared_network is None or np.random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            # Use shared network for action selection
            import torch
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                if torch.cuda.is_available():
                    state_tensor = state_tensor.cuda()
                q_values = self.shared_network(state_tensor)
                return q_values.argmax(1).item()
    
    def run_episode(self, max_steps: int = 10000, epsilon: float = 0.1):
        """Run a single episode and collect experiences"""
        obs, _ = self.env.reset()
        state = self.frame_stack.reset(obs, self.worker_id)
        episode_reward = 0
        episode_experiences = []
        steps = 0
        
        done = False
        while not done and steps < max_steps:
            # Select action
            action = self.select_action(state, epsilon)
            
            # Take step
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Process next state
            next_state = self.frame_stack.append(next_obs, self.worker_id)
            
            # Store experience
            experience = (state, action, reward, next_state, done, self.worker_id)
            episode_experiences.append(experience)
            
            # Update for next iteration
            state = next_state
            episode_reward += reward
            steps += 1
        
        # Add all experiences to replay buffer at once
        self.replay_buffer.push_batch(episode_experiences)
        
        self.episode_count += 1
        self.total_steps += steps
        self.episode_rewards.append(episode_reward)
        
        return {
            'worker_id': self.worker_id,
            'episode_reward': episode_reward,
            'episode_length': steps,
            'total_episodes': self.episode_count,
            'total_steps': self.total_steps
        }
    
    def run_continuous(self, max_episodes: int = None, epsilon_schedule=None):
        """Run episodes continuously"""
        self.running = True
        episode = 0
        
        while self.running and (max_episodes is None or episode < max_episodes):
            # Calculate epsilon for this episode
            if epsilon_schedule:
                epsilon = epsilon_schedule(episode)
            else:
                epsilon = max(0.01, 0.5 - episode * 0.001)  # Simple decay
            
            result = self.run_episode(epsilon=epsilon)
            episode += 1
            
            # Optional: brief pause to prevent overwhelming
            time.sleep(0.001)
        
        return self.get_statistics()
    
    def stop(self):
        """Stop the worker"""
        self.running = False
        self.frame_stack.clear_worker(self.worker_id)
    
    def get_statistics(self):
        """Get worker statistics"""
        return {
            'worker_id': self.worker_id,
            'total_episodes': self.episode_count,
            'total_steps': self.total_steps,
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'recent_avg_reward': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'max_reward': max(self.episode_rewards) if self.episode_rewards else 0,
            'min_reward': min(self.episode_rewards) if self.episode_rewards else 0
        }


class ParallelEnvironmentManager:
    """Manages multiple parallel environments for distributed RL"""
    
    def __init__(self, config, replay_buffer: DistributedReplayBuffer, 
                 num_workers: int = 4, shared_network=None):
        self.config = config
        self.num_workers = num_workers
        self.replay_buffer = replay_buffer
        self.shared_network = shared_network
        
        # Create workers
        self.workers = []
        for i in range(num_workers):
            worker = EnvironmentWorker(
                worker_id=i,
                env_name=config.env_name,
                frame_stack=config.frame_stack,
                replay_buffer=replay_buffer,
                shared_network=shared_network
            )
            self.workers.append(worker)
        
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.running = False
        self.future_to_worker = {}
        
    def start_collection(self, episodes_per_worker: int = None, epsilon_schedule=None):
        """Start parallel experience collection"""
        self.running = True
        
        # Submit jobs for all workers
        for worker in self.workers:
            future = self.executor.submit(
                worker.run_continuous, 
                episodes_per_worker, 
                epsilon_schedule
            )
            self.future_to_worker[future] = worker
        
        print(f"Started {self.num_workers} workers for parallel experience collection")
    
    def collect_batch(self, num_episodes_per_worker: int = 10, epsilon: float = 0.1):
        """Collect a specific number of episodes from each worker"""
        futures = []
        
        for worker in self.workers:
            for _ in range(num_episodes_per_worker):
                future = self.executor.submit(worker.run_episode, 10000, epsilon)
                futures.append(future)
        
        # Collect results
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Worker error: {e}")
        
        return results
    
    def update_shared_network(self, network):
        """Update the shared network for all workers"""
        self.shared_network = network
        for worker in self.workers:
            worker.shared_network = network
    
    def stop_collection(self):
        """Stop all workers"""
        self.running = False
        
        # Stop all workers
        for worker in self.workers:
            worker.stop()
        
        # Cancel pending futures
        for future in self.future_to_worker:
            future.cancel()
        
        # Wait for completion with timeout
        try:
            for future in as_completed(self.future_to_worker, timeout=5):
                try:
                    future.result()
                except Exception as e:
                    print(f"Worker shutdown error: {e}")
        except:
            pass
    
    def get_statistics(self):
        """Get aggregated statistics from all workers"""
        worker_stats = [worker.get_statistics() for worker in self.workers]
        
        total_episodes = sum(stats['total_episodes'] for stats in worker_stats)
        total_steps = sum(stats['total_steps'] for stats in worker_stats)
        
        all_rewards = []
        for worker in self.workers:
            all_rewards.extend(worker.episode_rewards)
        
        return {
            'num_workers': self.num_workers,
            'total_episodes': total_episodes,
            'total_steps': total_steps,
            'avg_reward_across_workers': np.mean([stats['avg_reward'] for stats in worker_stats if stats['avg_reward'] > 0]),
            'overall_avg_reward': np.mean(all_rewards) if all_rewards else 0,
            'overall_recent_avg': np.mean(all_rewards[-1000:]) if len(all_rewards) >= 1000 else np.mean(all_rewards) if all_rewards else 0,
            'worker_stats': worker_stats,
            'replay_buffer_stats': self.replay_buffer.get_statistics()
        }
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_collection()
        self.executor.shutdown(wait=True)