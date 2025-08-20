import math
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class MinMaxStats:
    """Tracks min and max values for value normalization"""
    minimum: float = float('inf')
    maximum: float = float('-inf')
    
    def update(self, value: float):
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        
    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


@dataclass
class Node:
    """Node in the MCTS tree"""
    prior: float
    hidden_state: Optional[torch.Tensor] = None
    reward: float = 0
    visit_count: int = 0
    value_sum: float = 0
    children: Dict[int, 'Node'] = field(default_factory=dict)
    
    def expanded(self) -> bool:
        return len(self.children) > 0
    
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def expand(self, actions: List[int], priors: np.ndarray, 
               reward: float, hidden_state: torch.Tensor):
        """Expand node with children"""
        self.reward = reward
        self.hidden_state = hidden_state
        for action, prior in zip(actions, priors):
            self.children[action] = Node(prior=prior)
    
    def add_exploration_noise(self, dirichlet_alpha: float, exploration_fraction: float):
        """Add Dirichlet noise to priors for exploration"""
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        for a, n in zip(actions, noise):
            self.children[a].prior = (
                self.children[a].prior * (1 - exploration_fraction) +
                n * exploration_fraction
            )


class MCTS:
    """Monte Carlo Tree Search for MuZero"""
    
    def __init__(self, config):
        self.config = config
        self.num_simulations = config.num_simulations
        self.action_space_size = config.action_space_size
        self.discount = config.discount
        
        # UCB constants
        self.pb_c_base = config.pb_c_base
        self.pb_c_init = config.pb_c_init
        
        # Exploration noise
        self.root_dirichlet_alpha = config.root_dirichlet_alpha
        self.root_exploration_fraction = config.root_exploration_fraction
        
    def run(self, observation: torch.Tensor, network: torch.nn.Module, 
            temperature: float = 1.0, add_exploration_noise: bool = True) -> Dict:
        """
        Run MCTS simulations and return action probabilities
        
        Args:
            observation: Current observation
            network: MuZero network for inference
            temperature: Temperature for action selection
            add_exploration_noise: Whether to add Dirichlet noise at root
            
        Returns:
            Dictionary with selected action, action probabilities, and tree statistics
        """
        with torch.no_grad():
            # Initial inference for root
            root_output = network.initial_inference(observation.unsqueeze(0))
            
            # Initialize root node
            root = Node(prior=0)
            
            # Expand root with initial policy
            policy_logits = root_output['policy_logits'].squeeze(0)
            policy = torch.softmax(policy_logits, dim=0).cpu().numpy()
            actions = list(range(self.action_space_size))
            root.expand(
                actions=actions,
                priors=policy,
                reward=0,
                hidden_state=root_output['state']
            )
            
            # Add exploration noise to root
            if add_exploration_noise:
                root.add_exploration_noise(
                    self.root_dirichlet_alpha,
                    self.root_exploration_fraction
                )
            
            # Track min-max statistics for value normalization
            min_max_stats = MinMaxStats()
            
            # Run simulations
            for _ in range(self.num_simulations):
                self._simulate(root, network, min_max_stats)
            
            # Extract action probabilities
            visit_counts = np.array([
                root.children[a].visit_count if a in root.children else 0
                for a in range(self.action_space_size)
            ])
            
            # Apply temperature
            if temperature == 0:
                # Deterministic: choose most visited action
                action = np.argmax(visit_counts)
                action_probs = np.zeros(self.action_space_size)
                action_probs[action] = 1.0
            else:
                # Stochastic: sample according to visit counts
                visit_counts = visit_counts ** (1 / temperature)
                action_probs = visit_counts / visit_counts.sum()
                action = np.random.choice(self.action_space_size, p=action_probs)
            
            return {
                'action': action,
                'action_probs': action_probs,
                'value': root.value(),
                'visit_counts': visit_counts,
                'root_value': root_output['value'].item()
            }
    
    def _simulate(self, root: Node, network: torch.nn.Module, min_max_stats: MinMaxStats):
        """Run a single simulation from root to leaf"""
        path = [root]
        actions = []
        node = root
        
        # Selection: traverse tree until we reach unexpanded node
        while node.expanded():
            action, node = self._select_child(node, min_max_stats)
            path.append(node)
            actions.append(action)
        
        # Get parent node for expansion
        parent = path[-2] if len(path) >= 2 else root
        
        # Expansion: expand the leaf node if it's been visited
        if node.visit_count > 0:
            # We need the hidden state from parent to expand
            with torch.no_grad():
                # Recurrent inference from parent state
                parent_state = parent.hidden_state
                action_tensor = torch.tensor([actions[-1]], device=parent_state.device)
                output = network.recurrent_inference(parent_state, action_tensor)
                
                # Expand node
                policy_logits = output['policy_logits'].squeeze(0)
                policy = torch.softmax(policy_logits, dim=0).cpu().numpy()
                node.expand(
                    actions=list(range(self.action_space_size)),
                    priors=policy,
                    reward=output['reward'].item(),
                    hidden_state=output['state']
                )
                
                value = output['value'].item()
        else:
            # Evaluation: use parent's prediction for this node
            value = 0  # Leaf node value
        
        # Backpropagation: update value estimates along path
        self._backpropagate(path, value, min_max_stats)
    
    def _select_child(self, node: Node, min_max_stats: MinMaxStats) -> Tuple[int, Node]:
        """Select child using UCB formula"""
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in node.children.items():
            score = self._ucb_score(node, child, min_max_stats)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def _ucb_score(self, parent: Node, child: Node, min_max_stats: MinMaxStats) -> float:
        """Calculate UCB score for child selection"""
        # Exploration term
        pb_c = math.log(
            (parent.visit_count + self.pb_c_base + 1) / self.pb_c_base
        ) + self.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
        
        # Prior score
        prior_score = pb_c * child.prior
        
        # Value score (normalized)
        if child.visit_count > 0:
            value_score = min_max_stats.normalize(child.value())
        else:
            value_score = 0
        
        return prior_score + value_score
    
    def _backpropagate(self, path: List[Node], value: float, min_max_stats: MinMaxStats):
        """Backpropagate value through path"""
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            min_max_stats.update(node.value())
            
            # Apply discount and add reward for next iteration
            value = node.reward + self.discount * value


class BatchedMCTS:
    """Batched MCTS for parallel tree search across multiple games"""
    
    def __init__(self, config):
        self.config = config
        self.mcts = MCTS(config)
    
    def run_batch(self, observations: torch.Tensor, network: torch.nn.Module,
                  temperature: float = 1.0, add_exploration_noise: bool = True) -> List[Dict]:
        """
        Run MCTS for a batch of observations
        
        Args:
            observations: Batch of observations (batch_size, ...)
            network: MuZero network
            temperature: Temperature for action selection
            add_exploration_noise: Whether to add exploration noise
            
        Returns:
            List of MCTS results for each observation
        """
        results = []
        
        # Run MCTS for each observation
        # Note: Could be parallelized for better performance
        for obs in observations:
            result = self.mcts.run(
                obs, network, temperature, add_exploration_noise
            )
            results.append(result)
        
        return results