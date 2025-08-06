import numpy as np
import random
from collections import deque
from typing import List, Tuple, Optional
try:
    from .sum_tree import SumTree
    from .data_buffer import PrioritizedReplayBuffer
except ImportError:
    from sum_tree import SumTree
    from data_buffer import PrioritizedReplayBuffer


class SequenceReplayBuffer:
    """
    Sequence-based replay buffer for R2D2
    
    Stores sequences of transitions instead of single transitions.
    Extends the prioritized replay concept to sequences.
    """
    
    def __init__(self, capacity: int, sequence_length: int = 80, burn_in_length: int = 40,
                 alpha: float = 0.6, beta: float = 0.4, epsilon: float = 1e-6,
                 priority_type: str = 'td_error'):
        """
        Args:
            capacity: Maximum number of sequences to store
            sequence_length: Length of each sequence (typically 80)
            burn_in_length: Length of burn-in period for LSTM warm-up (typically 40)
            alpha, beta, epsilon: Prioritized replay parameters
            priority_type: How to calculate priorities ('td_error', 'reward', 'random')
        """
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.burn_in_length = burn_in_length
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.priority_type = priority_type
        
        # Use sum tree for efficient priority-based sampling
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        
        # Temporary storage for building sequences
        self.episode_buffer = []
        self.episode_rewards = []
        self.episode_dones = []
        
    def push_transition(self, state, action, reward, next_state, done):
        """
        Add a single transition to the current episode buffer
        
        Args:
            state: Stacked frames from FrameStack (your existing implementation)
            action: Action taken
            reward: Reward received
            next_state: Next stacked frames
            done: Episode termination flag
        """
        transition = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        
        self.episode_buffer.append(transition)
        self.episode_rewards.append(reward)
        self.episode_dones.append(done)
        
        # Generate sequences as we go (every sequence_length//2 steps for overlap)
        if len(self.episode_buffer) >= self.sequence_length:
            # Check if we should generate a new sequence
            steps_since_last = len(self.episode_buffer) - self.sequence_length
            if steps_since_last % (self.sequence_length // 2) == 0:
                self._generate_sequence(steps_since_last)
        
        # If episode is done, process any remaining sequences and clean up
        if done:
            self._process_remaining_sequences()
            self._reset_episode_buffer()
    
    def _generate_sequence(self, start_idx):
        """Generate a single sequence starting at start_idx"""
        if start_idx + self.sequence_length > len(self.episode_buffer):
            return
            
        # Extract sequence
        sequence = self.episode_buffer[start_idx:start_idx + self.sequence_length]
        
        # Convert to arrays
        states = np.array([t['state'] for t in sequence])
        actions = np.array([t['action'] for t in sequence])
        rewards = np.array([t['reward'] for t in sequence])
        next_states = np.array([t['next_state'] for t in sequence])
        dones = np.array([t['done'] for t in sequence])
        
        # Create sequence data
        sequence_data = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'sequence_length': self.sequence_length,
            'burn_in_length': self.burn_in_length
        }
        
        # Calculate initial priority
        priority = self._calculate_initial_priority(sequence_data)
        
        # Add to replay buffer
        self.tree.add(priority, sequence_data)
        self.max_priority = max(self.max_priority, priority)
    
    def _process_remaining_sequences(self):
        """Process any remaining sequences at episode end"""
        if len(self.episode_buffer) < self.sequence_length:
            return
            
        # Generate any remaining sequences that weren't created during the episode
        last_generated = ((len(self.episode_buffer) - self.sequence_length) // (self.sequence_length // 2)) * (self.sequence_length // 2)
        
        # Create final sequences if there are enough remaining steps
        for start_idx in range(last_generated + self.sequence_length // 2, 
                             len(self.episode_buffer) - self.sequence_length + 1, 
                             self.sequence_length // 2):
            self._generate_sequence(start_idx)
    
    def _process_episode(self):
        """
        Legacy method - kept for compatibility but now unused
        """
        pass
    
    def _reset_episode_buffer(self):
        """Reset episode buffer for next episode"""
        self.episode_buffer = []
        self.episode_rewards = []
        self.episode_dones = []
    
    def _calculate_initial_priority(self, sequence_data):
        """Calculate initial priority for a sequence"""
        if self.priority_type == 'reward':
            # Use sum of absolute rewards in sequence
            return abs(sequence_data['rewards'].sum()) + self.epsilon
        elif self.priority_type == 'random':
            return random.random() + self.epsilon
        else:  # 'td_error' or default
            # Start with max priority, will be updated during training
            return self.max_priority
    
    def sample(self, batch_size: int):
        """
        Sample a batch of sequences
        
        Returns:
            batch: List of sequence dictionaries
            weights: Importance sampling weights
            idxs: Tree indices for priority updates
        """
        if self.tree.n_entries < batch_size:
            raise ValueError(f"Not enough sequences in buffer: {self.tree.n_entries} < {batch_size}")
        
        # Sample sequences using prioritized sampling
        sequences, tree_idxs, priority_values = self.tree.sample(batch_size)
        
        # Calculate importance sampling weights
        total = self.tree.total()
        weights = []
        
        for priority in priority_values:
            if priority > 0:
                prob = priority / total
                weight = (self.capacity * prob) ** (-self.beta)
                weights.append(weight)
            else:
                weights.append(1.0)
        
        # Normalize weights
        max_weight = max(weights) if weights else 1.0
        weights = [w / max_weight for w in weights]
        
        return sequences, np.array(weights), tree_idxs
    
    def update_priorities(self, idxs, priorities):
        """Update priorities for sequences based on TD errors"""
        for idx, priority in zip(idxs, priorities):
            # Apply prioritization exponent
            priority = abs(priority) + self.epsilon
            priority = priority ** self.alpha
            
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def update_beta(self, new_beta):
        """Update importance sampling correction factor"""
        self.beta = new_beta
    
    def __len__(self):
        return self.tree.n_entries
    
    def get_current_episode_length(self):
        """Get length of current episode being built"""
        return len(self.episode_buffer)


class SequenceDataLoader:
    """
    Helper class to convert sequences into training batches
    Handles burn-in and target sequence separation
    """
    
    @staticmethod
    def prepare_training_batch(sequences: List[dict]):
        """
        Prepare sequences for R2D2 training
        
        Args:
            sequences: List of sequence dictionaries from buffer
            
        Returns:
            burn_in_data: Data for LSTM warm-up (not used for loss)
            target_data: Data for training (used for loss calculation)
        """
        batch_size = len(sequences)
        
        if batch_size == 0:
            return None, None
        
        # Get dimensions from first sequence
        seq_length = sequences[0]['sequence_length']
        burn_in_length = sequences[0]['burn_in_length']
        target_length = seq_length - burn_in_length
        
        state_shape = sequences[0]['states'][0].shape
        
        # Prepare burn-in data (for LSTM initialization)
        burn_in_states = np.zeros((batch_size, burn_in_length) + state_shape)
        burn_in_actions = np.zeros((batch_size, burn_in_length), dtype=np.int64)
        
        # Prepare target data (for training)
        target_states = np.zeros((batch_size, target_length) + state_shape)
        target_actions = np.zeros((batch_size, target_length), dtype=np.int64)
        target_rewards = np.zeros((batch_size, target_length))
        target_next_states = np.zeros((batch_size, target_length) + state_shape)
        target_dones = np.zeros((batch_size, target_length), dtype=bool)
        
        # Fill arrays
        for i, seq in enumerate(sequences):
            # Burn-in data
            burn_in_states[i] = seq['states'][:burn_in_length]
            burn_in_actions[i] = seq['actions'][:burn_in_length]
            
            # Target data
            target_states[i] = seq['states'][burn_in_length:]
            target_actions[i] = seq['actions'][burn_in_length:]
            target_rewards[i] = seq['rewards'][burn_in_length:]
            target_next_states[i] = seq['next_states'][burn_in_length:]
            target_dones[i] = seq['dones'][burn_in_length:]
        
        burn_in_data = {
            'states': burn_in_states,
            'actions': burn_in_actions
        }
        
        target_data = {
            'states': target_states,
            'actions': target_actions,
            'rewards': target_rewards,
            'next_states': target_next_states,
            'dones': target_dones
        }
        
        return burn_in_data, target_data