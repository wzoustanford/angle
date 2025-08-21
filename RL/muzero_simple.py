#!/usr/bin/env python3
"""
Simple MuZero Implementation
============================
A clean, single-file implementation of the MuZero algorithm.
MuZero learns a model of the environment and uses it for planning with MCTS.

Key Components:
1. Representation network: h(observation) -> hidden_state
2. Dynamics network: g(hidden_state, action) -> next_hidden_state, reward
3. Prediction network: f(hidden_state) -> policy, value
4. MCTS planning in learned latent space
5. Self-play for data generation
6. Training with unrolled predictions
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional
import math
import random 

# Data structures
Experience = namedtuple('Experience', 
    ['observation', 'action', 'reward', 'search_policy', 'value_target'])
Trajectory = List[Experience]


class MuZeroNode:
    """
    Node in the MCTS tree.
    Stores statistics for action selection and value backup.
    """
    
    def __init__(self, prior: float = 0.0):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0.0
        self.children = {}
        self.hidden_state = None
        self.reward = 0.0
        self.is_expanded = False
        
    def value(self) -> float:
        """Average value of this node"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, parent_visits: int, c_puct: float = 1.25) -> float:
        """
        UCB score for action selection.
        Balances exploitation (value) and exploration (prior + visit count).
        """
        if self.visit_count == 0:
            # Unvisited nodes have maximum exploration bonus
            return float('inf')
        
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.value() + exploration
    
    def expand(self, priors: Dict[int, float], hidden_state: torch.Tensor):
        """
        Expand node by adding children for each action.
        """
        self.is_expanded = True
        self.hidden_state = hidden_state
        for action, prior in priors.items():
            self.children[action] = MuZeroNode(prior)


class MuZeroNetworks(nn.Module):
    """
    Neural networks for MuZero:
    - Representation: observation -> hidden state
    - Dynamics: (hidden state, action) -> (next hidden state, reward)
    - Prediction: hidden state -> (policy, value)
    """
    
    def __init__(self, 
                 observation_shape: Tuple[int, ...],
                 action_space_size: int,
                 hidden_size: int = 256,
                 representation_size: int = 256):
        super().__init__()
        
        self.action_space_size = action_space_size
        self.hidden_size = hidden_size
        
        # Representation network: h(observation) -> hidden_state
        # Simple CNN for image observations
        if len(observation_shape) == 3:  # Image input
            channels = observation_shape[0]
            self.representation = nn.Sequential(
                nn.Conv2d(channels, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((6, 6)),
                nn.Flatten(),
                nn.Linear(64 * 6 * 6, representation_size),
                nn.ReLU(),
                nn.Linear(representation_size, hidden_size)
            )
        else:  # Vector input
            input_size = np.prod(observation_shape)
            self.representation = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            )
        
        # Dynamics network: g(hidden_state, action) -> (next_hidden_state, reward)
        self.dynamics_state = nn.Sequential(
            nn.Linear(hidden_size + action_space_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.dynamics_reward = nn.Sequential(
            nn.Linear(hidden_size + action_space_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Prediction network: f(hidden_state) -> (policy, value)
        self.prediction_policy = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_space_size)
        )
        
        self.prediction_value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def initial_inference(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initial inference from observation.
        Returns: (hidden_state, policy_logits, value)
        """
        hidden_state = self.representation(observation)
        policy_logits = self.prediction_policy(hidden_state)
        value = self.prediction_value(hidden_state)
        return hidden_state, policy_logits, value
    
    def recurrent_inference(self, hidden_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Recurrent inference from hidden state and action.
        Returns: (next_hidden_state, reward, policy_logits, value)
        """
        # One-hot encode action
        action_one_hot = F.one_hot(action, self.action_space_size).float()
        
        # Concatenate hidden state and action
        state_action = torch.cat([hidden_state, action_one_hot], dim=-1)
        
        # Predict next state and reward
        next_hidden_state = self.dynamics_state(state_action)
        reward = self.dynamics_reward(state_action)
        
        # Predict policy and value for next state
        policy_logits = self.prediction_policy(next_hidden_state)
        value = self.prediction_value(next_hidden_state)
        
        return next_hidden_state, reward, policy_logits, value


class SimpleMuZero:
    """
    Simple MuZero implementation.
    Combines all components: networks, MCTS, self-play, and training.
    """
    
    def __init__(self,
                 observation_shape: Tuple[int, ...],
                 action_space_size: int,
                 max_moves: int = 512,
                 discount: float = 0.997,
                 num_simulations: int = 50,
                 batch_size: int = 128,
                 td_steps: int = 10,
                 num_unroll_steps: int = 5,
                 lr: float = 1e-3,
                 c_puct: float = 1.25,
                 dirichlet_alpha: float = 0.3,
                 exploration_fraction: float = 0.25,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        # Environment parameters
        self.observation_shape = observation_shape
        self.action_space_size = action_space_size
        self.max_moves = max_moves
        
        # Algorithm parameters
        self.discount = discount
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.td_steps = td_steps
        self.num_unroll_steps = num_unroll_steps
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_fraction = exploration_fraction
        self.device = device
        
        # Initialize networks
        self.network = MuZeroNetworks(
            observation_shape, 
            action_space_size
        ).to(device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=10000)
        
        # Training statistics
        self.training_step = 0
        
    def run_mcts(self, observation: np.ndarray, add_exploration_noise: bool = True) -> Tuple[np.ndarray, float]:
        """
        Run MCTS simulations from the given observation.
        Returns action probabilities based on visit counts and root value.
        
        Args:
            observation: Current observation
            add_exploration_noise: Whether to add Dirichlet noise at root (for self-play)
        """
        # Convert observation to tensor
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        # Initialize root with representation network
        with torch.no_grad():
            hidden_state, policy_logits, value = self.network.initial_inference(obs_tensor)
            
        # Create root node
        root = MuZeroNode()
        
        # Expand root with initial policy
        policy = F.softmax(policy_logits, dim=-1).squeeze().cpu().numpy()
        priors = {a: policy[a] for a in range(self.action_space_size)}
        
        # Add Dirichlet exploration noise at root (only during self-play)
        if add_exploration_noise:
            # Sample Dirichlet noise using configured alpha
            noise = np.random.dirichlet([self.dirichlet_alpha] * self.action_space_size)
            
            # Mix noise with priors: P(s,a) = (1-ε)p + ε*η
            for a in range(self.action_space_size):
                priors[a] = (1 - self.exploration_fraction) * priors[a] + self.exploration_fraction * noise[a]
        
        root.expand(priors, hidden_state)
        
        # Run simulations
        for _ in range(self.num_simulations):
            self._simulate(root)
        
        # Compute root value (average value from all simulations)
        root_value = root.value()
        
        # Return action probabilities based on visit counts
        visits = np.array([
            root.children[a].visit_count if a in root.children else 0
            for a in range(self.action_space_size)
        ])
        
        # Temperature could be added here for exploration
        if visits.sum() == 0:
            return np.ones(self.action_space_size) / self.action_space_size, 0.0
        
        return visits / visits.sum(), root_value
    
    def _simulate(self, root: MuZeroNode):
        """
        Run a single MCTS simulation.
        Traverse tree, expand leaf, and backup values.
        """
        node = root
        search_path = [root]
        
        # Selection: traverse tree until we reach unexpanded node
        current_depth = 0
        while node.is_expanded and current_depth < self.max_moves:
            # Select action with highest UCB score
            action, child = self._select_action(node)
            
            # If child hasn't been expanded yet, expand it
            if not child.is_expanded:
                # Use dynamics network to get next state
                with torch.no_grad():
                    parent_state = node.hidden_state
                    action_tensor = torch.LongTensor([action]).to(self.device)
                    
                    next_state, reward, policy_logits, value = \
                        self.network.recurrent_inference(parent_state, action_tensor)
                    
                    # Store reward prediction
                    child.reward = reward.item()
                    
                    # Expand child with predicted policy
                    policy = F.softmax(policy_logits, dim=-1).squeeze().cpu().numpy()
                    priors = {a: policy[a] for a in range(self.action_space_size)}
                    child.expand(priors, next_state)
                    
                    # Bootstrap value for backup
                    bootstrap_value = value.item()
                
                search_path.append(child)
                break
            
            node = child
            search_path.append(child)
            current_depth += 1
        
        # Backup the value through the search path
        # Note: bootstrap_value is only set if we expanded a new node
        if current_depth < self.max_moves and 'bootstrap_value' in locals():
            # We expanded a new leaf, use its value prediction
            self._backup(search_path, bootstrap_value)
        else:
            # Reached max depth with fully expanded tree
            # Must evaluate the final node's value (not 0!)
            # The value network estimates all future rewards from this position
            final_node = search_path[-1]
            with torch.no_grad():
                # Use prediction network to get value estimate
                hidden_state = final_node.hidden_state
                value = self.network.prediction_value(hidden_state)
                bootstrap_value = value.item()
            self._backup(search_path, bootstrap_value)
    
    def _select_action(self, node: MuZeroNode) -> Tuple[int, MuZeroNode]:
        """
        Select child with highest UCB score.
        """
        best_action = None
        best_child = None
        best_ucb = -float('inf')
        
        for action, child in node.children.items():
            ucb = child.ucb_score(node.visit_count, self.c_puct)
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def _backup(self, search_path: List[MuZeroNode], value: float):
        """
        Backup value through the search path.
        Update visit counts and value sums.
        """
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            # Include reward and discount for next step
            value = node.reward + self.discount * value
    
    def get_temperature(self) -> float:
        """
        Get temperature for action selection based on training steps.
        For Atari games, temperature is annealed:
        - Training steps < 500K: τ = 1.0
        - Training steps 500K-750K: τ = 0.5
        - Training steps > 750K: τ = 0.25
        """
        if self.training_step < 500000:
            return 1.0
        elif self.training_step < 750000:
            return 0.5
        else:
            return 0.25
    
    def self_play_game(self, env) -> Trajectory:
        """
        Play a game using MCTS for action selection.
        Returns trajectory of experiences for training.
        """
        trajectory = []
        root_values = []  # Store MCTS root values for value target computation
        observation = env.reset()
        # Handle gymnasium's reset returning (observation, info)
        if isinstance(observation, tuple):
            observation = observation[0]
        
        # Get current temperature based on training steps
        temperature = self.get_temperature()
        
        for _ in range(self.max_moves):
            # Run MCTS to get action probabilities AND root value (with Dirichlet noise)
            search_policy, root_value = self.run_mcts(observation, add_exploration_noise=True)
            root_values.append(root_value)
            
            # Apply temperature to action selection
            if temperature == 0:
                # Greedy selection (though we never reach τ=0 in Atari)
                action = np.argmax(search_policy)
            else:
                # Apply temperature: pi^(1/τ) and renormalize
                if temperature != 1.0:
                    search_policy = np.power(search_policy, 1/temperature)
                    search_policy = search_policy / search_policy.sum()
                action = np.random.choice(self.action_space_size, p=search_policy)
            
            # Execute action in environment (gymnasium format)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store experience (value targets will be computed later)
            trajectory.append(Experience(
                observation=observation,
                action=action,
                reward=reward,
                search_policy=search_policy,
                value_target=0.0  # Will be computed with root values
            ))
            
            observation = next_observation
            
            if done:
                break
        
        # Compute proper value targets using MCTS root values
        trajectory = self._compute_value_targets_with_mcts(trajectory, root_values)
        
        return trajectory
    
    def train_step(self):
        """
        Perform one training step on a batch from replay buffer.
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # Sample batch of trajectories
        batch_trajectories = random.sample(self.replay_buffer, self.batch_size)
        
        # Prepare batch tensors
        total_loss = 0
        policy_loss = 0
        value_loss = 0
        reward_loss = 0
        
        for trajectory in batch_trajectories:
            # Sample random position in trajectory
            game_pos = random.randint(0, len(trajectory) - 1)
            
            # Get observation at this position
            observation = trajectory[game_pos].observation
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            
            # Initial inference
            hidden_state, policy_logits, value_pred = self.network.initial_inference(obs_tensor)
            
            # Compute losses for initial position
            target_policy = torch.FloatTensor(
                trajectory[game_pos].search_policy
            ).unsqueeze(0).to(self.device)
            
            # Use pre-computed value target from MCTS
            target_value = trajectory[game_pos].value_target
            target_value_tensor = torch.FloatTensor([target_value]).to(self.device)
            
            # Policy loss (KL divergence)
            policy_loss += F.cross_entropy(policy_logits, target_policy)
            
            # Value loss (MSE)
            value_loss += F.mse_loss(value_pred.squeeze(), target_value_tensor.squeeze())
            
            # Unroll for K steps
            for k in range(1, self.num_unroll_steps + 1):
                if game_pos + k >= len(trajectory):
                    break
                
                # Get action at position k-1
                action = trajectory[game_pos + k - 1].action
                action_tensor = torch.LongTensor([action]).to(self.device)
                
                # Recurrent inference
                hidden_state, reward_pred, policy_logits, value_pred = \
                    self.network.recurrent_inference(hidden_state, action_tensor)
                
                # Compute targets
                target_reward = torch.FloatTensor(
                    [trajectory[game_pos + k - 1].reward]
                ).to(self.device)
                
                target_policy = torch.FloatTensor(
                    trajectory[game_pos + k].search_policy
                ).unsqueeze(0).to(self.device)
                
                # Use pre-computed value target from MCTS
                target_value = trajectory[game_pos + k].value_target
                target_value_tensor = torch.FloatTensor([target_value]).to(self.device)
                
                # Accumulate losses
                reward_loss += F.mse_loss(reward_pred.squeeze(), target_reward.squeeze())
                policy_loss += F.cross_entropy(policy_logits, target_policy)
                value_loss += F.mse_loss(value_pred.squeeze(), target_value_tensor.squeeze())
        
        # Total loss
        total_loss = policy_loss + value_loss + reward_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
        self.optimizer.step()
        
        self.training_step += 1
        
        return {
            'total_loss': total_loss.item() / self.batch_size,
            'policy_loss': policy_loss.item() / self.batch_size,
            'value_loss': value_loss.item() / self.batch_size,
            'reward_loss': reward_loss.item() / self.batch_size,
            'training_step': self.training_step
        }
    
    def _compute_value_targets_with_mcts(self, trajectory: Trajectory, root_values: List[float]) -> Trajectory:
        """
        Compute n-step value targets using MCTS root values.
        This is the correct MuZero way - using search values for bootstrapping.
        """
        updated_trajectory = []
        
        for i, exp in enumerate(trajectory):
            # Compute n-step return with bootstrap from MCTS root value
            value_target = 0.0
            
            # Sum n-step rewards
            for k in range(self.td_steps):
                if i + k < len(trajectory):
                    value_target += (self.discount ** k) * trajectory[i + k].reward
            
            # Bootstrap from MCTS root value at position i+n
            bootstrap_index = i + self.td_steps
            if bootstrap_index < len(root_values):
                # Use the MCTS root value as bootstrap (THIS IS THE KEY FIX!)
                value_target += (self.discount ** self.td_steps) * root_values[bootstrap_index]
            else:
                # If we're near the end, use remaining rewards
                for k in range(self.td_steps, len(trajectory) - i):
                    if i + k < len(trajectory):
                        value_target += (self.discount ** k) * trajectory[i + k].reward
            
            # Create updated experience with proper value target
            updated_exp = Experience(
                observation=exp.observation,
                action=exp.action,
                reward=exp.reward,
                search_policy=exp.search_policy,
                value_target=value_target  # NOW using MCTS values!
            )
            updated_trajectory.append(updated_exp)
        
        return updated_trajectory
    
    def _compute_target_value(self, trajectory: Trajectory, index: int) -> float:
        """
        Compute n-step bootstrapped value target.
        V(s_t) = sum_{k=0}^{n-1} gamma^k * r_{t+k} + gamma^n * V(s_{t+n})
        """
        bootstrap_index = index + self.td_steps
        
        if bootstrap_index < len(trajectory):
            # Bootstrap from value at future position
            # In practice, we'd use the value from MCTS or network
            # For simplicity, using discounted sum of rewards
            value = 0.0
            for i in range(index, min(bootstrap_index, len(trajectory))):
                value += (self.discount ** (i - index)) * trajectory[i].reward
            
            # Add bootstrapped value if not terminal
            if bootstrap_index < len(trajectory):
                # Here we'd ideally use the value from search
                # For now, continue with reward sum
                remaining_value = 0.0
                for i in range(bootstrap_index, len(trajectory)):
                    remaining_value += (self.discount ** (i - bootstrap_index)) * trajectory[i].reward
                value += (self.discount ** self.td_steps) * remaining_value
        else:
            # Use Monte Carlo return
            value = 0.0
            for i in range(index, len(trajectory)):
                value += (self.discount ** (i - index)) * trajectory[i].reward
        
        return value
    
    def update_replay_buffer(self, trajectory: Trajectory):
        """
        Add trajectory to replay buffer.
        """
        self.replay_buffer.append(trajectory)
    
    def save_checkpoint(self, path: str):
        """
        Save model checkpoint.
        """
        torch.save({
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'training_step': self.training_step
        }, path)
    
    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.training_step = checkpoint['training_step']


def main():
    """
    Example usage of SimpleMuZero.
    """
    import gymnasium
    
    # Create environment
    env = gymnasium.make('CartPole-v1')
    
    # Get environment specs
    observation_shape = env.observation_space.shape
    action_space_size = env.action_space.n
    
    # Initialize MuZero
    muzero = SimpleMuZero(
        observation_shape=observation_shape,
        action_space_size=action_space_size,
        num_simulations=25,  # Fewer simulations for CartPole
        batch_size=32,
        lr=1e-3
    )
    
    print("Starting MuZero training on CartPole...")
    print(f"Observation shape: {observation_shape}")
    print(f"Action space size: {action_space_size}")
    print(f"Device: {muzero.device}")
    
    # Training loop
    for episode in range(100):
        # Self-play
        trajectory = muzero.self_play_game(env)
        muzero.update_replay_buffer(trajectory)
        
        # Training
        if episode > 10:  # Start training after collecting some data
            for _ in range(10):  # Multiple training steps per episode
                losses = muzero.train_step()
                
        if episode % 1 == 0:
            total_reward = sum(exp.reward for exp in trajectory)
            print(f"Episode {episode}: Reward={total_reward:.1f}, Steps={len(trajectory)}")
            if episode > 10 and losses:
                print(f"  Losses - Total: {losses['total_loss']:.4f}, "
                      f"Policy: {losses['policy_loss']:.4f}, "
                      f"Value: {losses['value_loss']:.4f}")
    
    print("Training complete!")
    muzero.save_checkpoint('muzero_cartpole.pt')


if __name__ == "__main__":
    main()