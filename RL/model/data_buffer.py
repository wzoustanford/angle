import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """Experience replay buffer for DQN training"""
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done))
    
    def __len__(self):
        return len(self.buffer)


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