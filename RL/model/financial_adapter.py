"""
Adapter to bridge financial environments with DQN/R2D2 infrastructure.
This allows using angle_rl's financial environments with angle/RL's training code.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from typing import Dict, Tuple, Optional, Any

# Add angle_rl to path
sys.path.append('/home/ubuntu/code/angle_rl/invest')

# Import original models from angle_rl
from model.policy_model import PolicyModel
from model.value_model import ValueModel


class FinancialStateAdapter:
    """
    Adapter that converts between flat observations and financial state dictionaries.
    """
    
    def __init__(self, num_tickers: int, feature_dim: int = 249, hidden_dim: int = 47):
        self.num_tickers = num_tickers
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Calculate offsets for flat observation components
        self.delta_start = 0
        self.delta_end = num_tickers
        
        self.action_start = self.delta_end
        self.action_end = self.action_start + num_tickers
        
        self.sharpe_idx = self.action_end
        self.x_idx = self.sharpe_idx + 1
        
        self.acts_start = self.x_idx + 1
        self.acts_end = self.acts_start + hidden_dim
        
        self.features_start = self.acts_end
        self.features_end = self.features_start + feature_dim
        
        self.total_dim = self.features_end
    
    def flat_to_state_dict(self, flat_obs: np.ndarray) -> Dict:
        """
        Convert flat observation to state dictionary.
        
        Args:
            flat_obs: Flat observation array from DQN environment
            
        Returns:
            State dictionary compatible with PolicyModel
        """
        if isinstance(flat_obs, torch.Tensor):
            flat_obs = flat_obs.numpy()
        
        state = {
            'delta': torch.from_numpy(flat_obs[self.delta_start:self.delta_end]).float().unsqueeze(0),
            'action': torch.from_numpy(flat_obs[self.action_start:self.action_end]).float().unsqueeze(0),
            'sharpe': torch.tensor([flat_obs[self.sharpe_idx]]).float().view(1, 1),
            'X': float(flat_obs[self.x_idx]),
            'policy_pooled_acts': torch.from_numpy(flat_obs[self.acts_start:self.acts_end]).float().unsqueeze(0),
            'features': None,  # Will be reconstructed if needed
            'tickers': None,   # Will use dummy tickers
            'prices': torch.zeros((self.num_tickers, 1))  # Placeholder
        }
        
        # Reconstruct features if available
        if self.features_end <= len(flat_obs):
            feature_summary = flat_obs[self.features_start:self.features_end]
            # Expand summary to all stocks (simple broadcast)
            features = torch.from_numpy(feature_summary).float().unsqueeze(0).repeat(self.num_tickers, 1)
            state['features'] = features
            state['tickers'] = [f'STOCK_{i}' for i in range(self.num_tickers)]
        
        return state
    
    def state_dict_to_flat(self, state: Dict) -> np.ndarray:
        """
        Convert state dictionary to flat observation.
        
        Args:
            state: State dictionary from financial environment
            
        Returns:
            Flat observation array for DQN
        """
        components = []
        
        # Delta
        delta = state['delta'].squeeze()
        components.append(delta.numpy() if isinstance(delta, torch.Tensor) else delta)
        
        # Action
        action = state['action'].squeeze()
        components.append(action.numpy() if isinstance(action, torch.Tensor) else action)
        
        # Sharpe
        sharpe = state['sharpe'].squeeze()
        components.append(np.array([sharpe]) if np.isscalar(sharpe) else sharpe.numpy())
        
        # X
        components.append(np.array([state['X']]))
        
        # Policy pooled acts
        acts = state['policy_pooled_acts'].squeeze()
        components.append(acts.numpy() if isinstance(acts, torch.Tensor) else acts)
        
        # Features (mean summary)
        if state['features'] is not None:
            features = state['features']
            if isinstance(features, torch.Tensor):
                features = features.numpy()
            feature_summary = np.mean(features, axis=0)
            components.append(feature_summary)
        else:
            components.append(np.zeros(self.feature_dim))
        
        return np.concatenate(components).astype(np.float32)


class FinancialDQNNetwork(nn.Module):
    """
    DQN network that wraps the original PolicyModel for Q-value estimation.
    """
    
    def __init__(self, 
                 policy_model: PolicyModel,
                 num_actions: int,
                 state_adapter: FinancialStateAdapter,
                 use_dueling: bool = True):
        super().__init__()
        
        self.policy_model = policy_model
        self.num_actions = num_actions
        self.state_adapter = state_adapter
        self.use_dueling = use_dueling
        
        # Q-value heads (added on top of policy model)
        if use_dueling:
            self.value_head = nn.Linear(policy_model.hidden_dim, 1)
            self.advantage_head = nn.Linear(policy_model.hidden_dim, num_actions)
        else:
            self.q_head = nn.Linear(policy_model.hidden_dim, num_actions)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass converting flat observation to Q-values.
        
        Args:
            x: Flat observation tensor (batch_size, obs_dim)
            
        Returns:
            Q-values for all actions (batch_size, num_actions)
        """
        batch_size = x.shape[0]
        q_values_list = []
        
        # Process each sample in batch
        for i in range(batch_size):
            # Convert flat obs to state dict
            flat_obs = x[i].cpu().numpy()
            state = self.state_adapter.flat_to_state_dict(flat_obs)
            
            # Get policy activations (using return_acts=True)
            with torch.no_grad():
                # We need the hidden activations, not the final output
                _, acts = self.policy_model(state, state['tickers'], return_acts=True)
                
                # Pool activations
                if acts.dim() > 1:
                    pooled_acts = torch.max(acts, dim=0)[0]
                else:
                    pooled_acts = acts
            
            # Compute Q-values from pooled activations
            if self.use_dueling:
                value = self.value_head(pooled_acts)
                advantage = self.advantage_head(pooled_acts)
                q_values = value + advantage - advantage.mean()
            else:
                q_values = self.q_head(pooled_acts)
            
            q_values_list.append(q_values)
        
        # Stack batch
        q_values = torch.stack(q_values_list)
        return q_values


class FinancialPolicyWrapper:
    """
    Wrapper that makes PolicyModel compatible with DQN training.
    """
    
    def __init__(self,
                 num_tickers: int,
                 num_actions: int = 100,
                 ticker_hash_file: str = None,
                 device: str = 'cpu'):
        
        self.num_tickers = num_tickers
        self.num_actions = num_actions
        self.device = torch.device(device)
        
        # Load ticker hash if provided
        if ticker_hash_file:
            import pickle
            with open(ticker_hash_file, 'rb') as f:
                ticker_data = pickle.load(f)
                self.shuffle_dict = ticker_data['hash_D']
                self.num_tickers = ticker_data['num_tickers']
        else:
            self.shuffle_dict = None
        
        # Create original policy model
        self.policy_model = PolicyModel(
            shuffle_dict=self.shuffle_dict,
            num_tickers=self.num_tickers,
            device=self.device
        ).to(self.device)
        
        # Create state adapter
        self.state_adapter = FinancialStateAdapter(num_tickers=self.num_tickers)
        
        # Create DQN network wrapper
        self.dqn_network = FinancialDQNNetwork(
            policy_model=self.policy_model,
            num_actions=num_actions,
            state_adapter=self.state_adapter,
            use_dueling=True
        ).to(self.device)
        
        # For compatibility with DQN interface
        self.network = self.dqn_network
    
    def get_q_values(self, observation: torch.Tensor) -> torch.Tensor:
        """Get Q-values for observation."""
        return self.dqn_network(observation)
    
    def select_action(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        """Select action using epsilon-greedy policy."""
        if np.random.random() < epsilon:
            return np.random.randint(0, self.num_actions)
        
        with torch.no_grad():
            obs_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
            q_values = self.get_q_values(obs_tensor)
            action = q_values.argmax().item()
        
        return action
    
    def decode_discrete_action(self, action: int) -> torch.Tensor:
        """
        Convert discrete action to continuous portfolio allocation.
        
        Args:
            action: Discrete action index
            
        Returns:
            Portfolio allocation tensor
        """
        portfolio = torch.zeros(self.num_tickers)
        
        # Simple decoding strategy
        if self.num_tickers <= 10:
            stock_idx = action % self.num_tickers
            weight_level = action // self.num_tickers
            weight = weight_level / 20.0  # 21 levels from 0 to 1
            portfolio[stock_idx] = weight
        else:
            # Focus on subset of stocks
            focus_size = min(20, self.num_tickers)
            focus_action = action % (focus_size * 21)
            stock_idx = focus_action % focus_size
            weight_level = focus_action // focus_size
            
            primary_weight = weight_level / 20.0
            portfolio[stock_idx] = primary_weight
            
            # Distribute remaining
            remaining = max(0, 1.0 - primary_weight)
            if remaining > 0 and focus_size > 1:
                other_weight = remaining / (focus_size - 1)
                for i in range(focus_size):
                    if i != stock_idx:
                        portfolio[i] = other_weight
        
        # Normalize
        total = torch.sum(portfolio)
        if total > 0:
            portfolio = portfolio / total
        else:
            portfolio = torch.ones(self.num_tickers) / self.num_tickers
        
        return portfolio


# Factory function
def create_financial_dqn_wrapper(num_tickers: int,
                                num_actions: int = 100,
                                ticker_hash_file: str = None,
                                device: str = 'cpu') -> FinancialPolicyWrapper:
    """
    Create financial DQN wrapper.
    
    Args:
        num_tickers: Number of stocks
        num_actions: Number of discrete actions
        ticker_hash_file: Path to ticker hash file
        device: Device to use
        
    Returns:
        FinancialPolicyWrapper instance
    """
    return FinancialPolicyWrapper(
        num_tickers=num_tickers,
        num_actions=num_actions,
        ticker_hash_file=ticker_hash_file,
        device=device
    )