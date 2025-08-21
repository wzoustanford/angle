import numpy as np
import random
from collections import deque
try:
    from .sum_tree import SumTree
except ImportError:
    from sum_tree import SumTree


class ReplayBuffer:
    """Experience replay buffer for DQN training"""
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        # Detach tensors to prevent memory leak from computation graph
        if hasattr(state, 'detach'):
            state = state.detach()
        if hasattr(next_state, 'detach'):
            next_state = next_state.detach()
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done))
    
    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer"""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, epsilon: float = 1e-6, priority_type: str = 'td_error'):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.beta = beta   # Importance sampling weight
        self.epsilon = epsilon  # Small constant to prevent zero priorities
        self.priority_type = priority_type
        
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer with max priority"""
        # Detach tensors to prevent memory leak from computation graph
        if hasattr(state, 'detach'):
            state = state.detach()
        if hasattr(next_state, 'detach'):
            next_state = next_state.detach()
        experience = (state, action, reward, next_state, done)
        priority = self.max_priority
        self.tree.add(priority, experience)
    
    def sample(self, batch_size: int):
        """Sample batch with priorities and importance weights"""
        batch = []
        idxs = []
        priorities = []
        
        # Sample experiences
        experiences, tree_idxs, priority_values = self.tree.sample(batch_size)
        
        for exp in experiences:
            batch.append(exp)
        
        idxs = tree_idxs
        priorities = priority_values
        
        # Calculate importance sampling weights
        total = self.tree.total()
        weights = []
        
        for i in range(batch_size):
            if priorities[i] > 0:
                prob = priorities[i] / total
                weight = (self.capacity * prob) ** (-self.beta)
                weights.append(weight)
            else:
                weights.append(1.0)
        
        # Normalize weights
        max_weight = max(weights)
        weights = [w / max_weight for w in weights]
        
        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones), 
                np.array(weights), idxs)
    
    def update_priorities(self, idxs, priorities):
        """Update priorities based on TD errors or other metrics"""
        for idx, priority in zip(idxs, priorities):
            # Apply prioritization exponent
            priority = abs(priority) + self.epsilon
            priority = priority ** self.alpha
            
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def calculate_priority(self, td_error=None, reward=None):
        """Calculate priority based on specified type"""
        if self.priority_type == 'td_error' and td_error is not None:
            return abs(td_error) + self.epsilon
        elif self.priority_type == 'reward' and reward is not None:
            return abs(reward) + self.epsilon
        elif self.priority_type == 'random':
            return random.random() + self.epsilon
        else:
            return 1.0 + self.epsilon
    
    def update_beta(self, new_beta):
        """Update importance sampling correction factor"""
        self.beta = new_beta
    
    def __len__(self):
        return self.tree.n_entries


class FrameStack:
    """Stack frames for temporal information"""
    def __init__(self, num_stack: int):
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
    
    def reset(self, observation):
        for _ in range(self.num_stack):
            self.frames.append(observation)
        return self._get_observation()
    
    def append(self, observation):
        self.frames.append(observation)
        return self._get_observation()
    
    def _get_observation(self):
        return np.array(self.frames).transpose(0, 3, 1, 2).reshape(-1, *self.frames[0].shape[:2])