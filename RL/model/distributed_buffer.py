import numpy as np
import random
import threading
from collections import deque
from typing import Tuple, List


class DistributedReplayBuffer:
    """Thread-safe replay buffer for collecting experiences from multiple parallel environments"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.lock = threading.Lock()
        self._episode_boundaries = []  # Track episode boundaries for better sampling
        
    def push(self, state, action, reward, next_state, done, worker_id: int = 0):
        """Add experience to buffer (thread-safe)"""
        with self.lock:
            self.buffer.append((state, action, reward, next_state, done, worker_id))
            if done:
                self._episode_boundaries.append(len(self.buffer) - 1)
                # Keep only recent episode boundaries
                if len(self._episode_boundaries) > 1000:
                    self._episode_boundaries = self._episode_boundaries[-500:]
    
    def push_batch(self, experiences: List[Tuple]):
        """Add multiple experiences at once (more efficient for distributed collection)"""
        with self.lock:
            for exp in experiences:
                state, action, reward, next_state, done, worker_id = exp
                self.buffer.append((state, action, reward, next_state, done, worker_id))
                if done:
                    self._episode_boundaries.append(len(self.buffer) - 1)
    
    def sample(self, batch_size: int, prioritize_recent: bool = False):
        """Sample experiences from buffer"""
        with self.lock:
            if len(self.buffer) < batch_size:
                return None
            
            if prioritize_recent and len(self.buffer) > batch_size * 2:
                # Sample more from recent experiences
                recent_size = min(len(self.buffer) // 2, batch_size)
                recent_indices = list(range(len(self.buffer) - recent_size, len(self.buffer)))
                old_indices = list(range(0, len(self.buffer) - recent_size))
                
                # 70% from recent, 30% from older experiences
                recent_samples = random.sample(recent_indices, int(batch_size * 0.7))
                old_samples = random.sample(old_indices, batch_size - len(recent_samples))
                indices = recent_samples + old_samples
                random.shuffle(indices)
                
                batch = [self.buffer[i] for i in indices]
            else:
                batch = random.sample(self.buffer, batch_size)
            
            # Unpack experiences
            states, actions, rewards, next_states, dones, worker_ids = zip(*batch)
            
            return (np.array(states), np.array(actions), np.array(rewards), 
                   np.array(next_states), np.array(dones), np.array(worker_ids))
    
    def sample_episode(self):
        """Sample a complete episode for analysis"""
        with self.lock:
            if not self._episode_boundaries:
                return None
            
            # Find a random complete episode
            end_idx = random.choice(self._episode_boundaries)
            start_idx = 0
            for boundary in reversed(self._episode_boundaries):
                if boundary < end_idx:
                    start_idx = boundary + 1
                    break
            
            episode = list(self.buffer)[start_idx:end_idx + 1]
            states, actions, rewards, next_states, dones, worker_ids = zip(*episode)
            
            return {
                'states': np.array(states),
                'actions': np.array(actions),
                'rewards': np.array(rewards),
                'next_states': np.array(next_states),
                'dones': np.array(dones),
                'worker_ids': np.array(worker_ids),
                'episode_length': len(episode),
                'total_reward': sum(rewards)
            }
    
    def get_statistics(self):
        """Get buffer statistics"""
        with self.lock:
            stats = {
                'size': len(self.buffer),
                'capacity': self.capacity,
                'fill_ratio': len(self.buffer) / self.capacity,
                'num_episodes': len(self._episode_boundaries)
            }
            
            if self.buffer:
                # Calculate worker distribution
                worker_counts = {}
                for _, _, _, _, _, worker_id in self.buffer:
                    worker_counts[worker_id] = worker_counts.get(worker_id, 0) + 1
                stats['worker_distribution'] = worker_counts
            
            return stats
    
    def clear(self):
        """Clear the buffer"""
        with self.lock:
            self.buffer.clear()
            self._episode_boundaries.clear()
    
    def __len__(self):
        return len(self.buffer)


class DistributedFrameStack:
    """Thread-safe frame stacking for distributed environments"""
    
    def __init__(self, num_stack: int):
        self.num_stack = num_stack
        self.worker_frames = {}  # Separate frame stacks per worker
        self.lock = threading.Lock()
    
    def reset(self, observation, worker_id: int = 0):
        """Reset frame stack for a specific worker"""
        with self.lock:
            frames = deque(maxlen=self.num_stack)
            for _ in range(self.num_stack):
                frames.append(observation)
            self.worker_frames[worker_id] = frames
            return self._get_observation(worker_id)
    
    def append(self, observation, worker_id: int = 0):
        """Add frame for a specific worker"""
        with self.lock:
            if worker_id not in self.worker_frames:
                # Initialize if not exists
                self.reset(observation, worker_id)
            else:
                self.worker_frames[worker_id].append(observation)
            return self._get_observation(worker_id)
    
    def _get_observation(self, worker_id: int):
        """Get stacked observation for worker"""
        frames = self.worker_frames[worker_id]
        return np.array(frames).transpose(0, 3, 1, 2).reshape(-1, *frames[0].shape[:2])
    
    def clear_worker(self, worker_id: int):
        """Clear frames for a specific worker"""
        with self.lock:
            if worker_id in self.worker_frames:
                del self.worker_frames[worker_id]