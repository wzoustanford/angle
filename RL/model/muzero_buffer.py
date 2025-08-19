import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import random


@dataclass
class GameHistory:
    """Stores the history of a single game"""
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    policies: List[np.ndarray] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    
    # For prioritized replay
    priorities: Optional[List[float]] = None
    game_priority: float = 1.0
    
    def __len__(self):
        return len(self.actions)
    
    def store(self, observation: np.ndarray, action: int, reward: float,
              policy: np.ndarray, value: float):
        """Store a transition"""
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.policies.append(policy)
        self.values.append(value)
    
    def make_target(self, state_index: int, num_unroll_steps: int,
                    td_steps: int, discount: float) -> Dict:
        """
        Create training target for a specific position
        
        Args:
            state_index: Position in game to create target for
            num_unroll_steps: Number of steps to unroll
            td_steps: Number of steps for n-step returns
            discount: Discount factor
            
        Returns:
            Dictionary with observation, actions, and targets
        """
        targets = {
            'observation': self.observations[state_index],
            'actions': [],
            'target_values': [],
            'target_rewards': [],
            'target_policies': []
        }
        
        # We need exactly num_unroll_steps actions and num_unroll_steps + 1 values/policies/rewards
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            # Bootstrap from value at td_steps or end of game
            bootstrap_index = current_index + td_steps
            
            if bootstrap_index < len(self.values):
                # n-step return with bootstrap
                value = self.values[bootstrap_index] * (discount ** td_steps)
            else:
                # No bootstrap, use 0
                value = 0
            
            # Add discounted rewards
            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * (discount ** i)
            
            if current_index < len(self.values):
                targets['target_values'].append(value)
                targets['target_policies'].append(self.policies[current_index])
                
                # Add reward (0 for first step)
                if current_index > state_index:
                    targets['target_rewards'].append(self.rewards[current_index - 1])
                else:
                    targets['target_rewards'].append(0)
            else:
                # Pad with zeros if we've reached the end
                targets['target_values'].append(0)
                targets['target_policies'].append(np.zeros_like(self.policies[0]))
                targets['target_rewards'].append(0)
        
        # Build actions list - exactly num_unroll_steps actions
        for i in range(num_unroll_steps):
            action_index = state_index + i
            if action_index < len(self.actions):
                targets['actions'].append(self.actions[action_index])
            else:
                # Pad with random valid action
                targets['actions'].append(0)
        
        return targets
    
    def compute_priority(self, state_index: int) -> float:
        """Compute priority for a specific position (for prioritized replay)"""
        # Simple priority based on value prediction error
        if state_index < len(self.values) - 1:
            # Could use TD error or other metrics
            return abs(self.values[state_index])
        return 1.0


class MuZeroReplayBuffer:
    """Replay buffer for MuZero training"""
    
    def __init__(self, config):
        self.config = config
        self.buffer_size = config.replay_buffer_size
        self.num_unroll_steps = config.num_unroll_steps
        self.td_steps = config.td_steps
        self.discount = config.discount
        self.batch_size = config.batch_size
        
        # Priority replay settings
        self.use_priority = config.use_priority_replay
        self.priority_alpha = config.priority_alpha
        self.priority_beta = config.priority_beta
        
        # Storage
        self.buffer: List[GameHistory] = []
        self.total_games = 0
        
    def save_game(self, game: GameHistory):
        """Save a game to the buffer"""
        if len(self.buffer) >= self.buffer_size:
            # Remove oldest game
            self.buffer.pop(0)
        
        # Compute priorities if using prioritized replay
        if self.use_priority:
            priorities = []
            for i in range(len(game)):
                priorities.append(game.compute_priority(i))
            game.priorities = priorities
            game.game_priority = max(priorities) if priorities else 1.0
        
        self.buffer.append(game)
        self.total_games += 1
    
    def sample_batch(self) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of training data
        
        Returns:
            Dictionary with batched tensors for training
        """
        # Sample games
        if self.use_priority:
            games = self._sample_games_prioritized()
        else:
            games = self._sample_games_uniform()
        
        # Sample positions from games
        batch_data = {
            'observations': [],
            'actions': [],
            'target_values': [],
            'target_rewards': [],
            'target_policies': [],
            'weights': []  # Importance sampling weights
        }
        
        for game, weight in games:
            # Sample random position from game
            game_len = len(game)
            if game_len > 0:
                position = np.random.randint(0, game_len)
                
                # Create target
                target = game.make_target(
                    position,
                    self.num_unroll_steps,
                    self.td_steps,
                    self.discount
                )
                
                # Add to batch
                batch_data['observations'].append(target['observation'])
                batch_data['actions'].append(target['actions'])
                batch_data['target_values'].append(target['target_values'])
                batch_data['target_rewards'].append(target['target_rewards'])
                batch_data['target_policies'].append(target['target_policies'])
                batch_data['weights'].append(weight)
        
        # Convert to tensors
        batch_tensors = {
            'observations': torch.FloatTensor(np.array(batch_data['observations'])),
            'actions': torch.LongTensor(batch_data['actions']),
            'target_values': torch.FloatTensor(batch_data['target_values']),
            'target_rewards': torch.FloatTensor(batch_data['target_rewards']),
            'target_policies': torch.FloatTensor(np.array(batch_data['target_policies'])),
            'weights': torch.FloatTensor(batch_data['weights'])
        }
        
        return batch_tensors
    
    def _sample_games_uniform(self) -> List[Tuple[GameHistory, float]]:
        """Sample games uniformly"""
        games = []
        for _ in range(self.batch_size):
            if self.buffer:
                game = random.choice(self.buffer)
                games.append((game, 1.0))  # Uniform weight
        return games
    
    def _sample_games_prioritized(self) -> List[Tuple[GameHistory, float]]:
        """Sample games with prioritization"""
        if not self.buffer:
            return []
        
        # Compute sampling probabilities
        priorities = np.array([game.game_priority for game in self.buffer])
        priorities = priorities ** self.priority_alpha
        probs = priorities / priorities.sum()
        
        # Sample games
        games = []
        indices = np.random.choice(len(self.buffer), self.batch_size, p=probs)
        
        # Compute importance sampling weights
        min_prob = probs.min()
        max_weight = (min_prob * len(self.buffer)) ** (-self.priority_beta)
        
        for idx in indices:
            game = self.buffer[idx]
            
            # Importance sampling weight
            weight = (probs[idx] * len(self.buffer)) ** (-self.priority_beta)
            weight = weight / max_weight  # Normalize
            
            games.append((game, weight))
        
        return games
    
    def update_priorities(self, game_indices: List[int], priorities: List[float]):
        """Update priorities for games (for prioritized replay)"""
        for idx, priority in zip(game_indices, priorities):
            if 0 <= idx < len(self.buffer):
                self.buffer[idx].game_priority = priority
    
    def __len__(self):
        return sum(len(game) for game in self.buffer)
    
    def is_ready(self) -> bool:
        """Check if buffer has enough data for training"""
        return len(self) >= self.batch_size