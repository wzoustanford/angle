import torch
import numpy as np
import pickle
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple
from .device_utils import get_device_manager


class FinancialEnvironment(gym.Env):
    """
    Financial trading environment that adapts financial data for DQN/R2D2 training.
    
    Observations: Market features + portfolio state
    Actions: Portfolio allocation weights (continuous, converted to discrete bins)
    Rewards: Sharpe ratio (risk-adjusted returns)
    """
    
    def __init__(self, 
                 data_list_file: str,
                 ticker_hash_file: str,
                 num_action_bins: int = 101,  # Discrete actions: 0%, 1%, 2%, ..., 100%
                 transaction_cost: float = 0.0015,
                 action_update_interval: int = 10,
                 max_episode_steps: int = 200,
                 device: str = None):
        super().__init__()
        
        self.devmgr = get_device_manager(device)
        self.device = self.devmgr.device
        
        # Load financial data
        self.data_list_file = data_list_file
        self.ticker_hash_file = ticker_hash_file
        self.data_files = self._load_data_list()
        self.ticker_hash = self._load_ticker_hash()
        
        # Environment parameters
        self.num_action_bins = num_action_bins
        self.transaction_cost = transaction_cost
        self.action_update_interval = action_update_interval
        self.max_episode_steps = max_episode_steps
        
        # State tracking
        self.current_step = 0
        self.current_data_idx = 0
        self.portfolio_value = 1.0
        self.previous_portfolio = None
        self.previous_prices = None
        self.episode_data = []
        
        # Gym spaces
        self.num_stocks = self.ticker_hash['num_tickers']
        
        # Observation: features (249) + portfolio state (num_stocks) + scalar values (3)
        self.feature_dim = 249  # From data analysis
        obs_dim = self.feature_dim + self.num_stocks + 3  # +3 for portfolio_value, prev_sharpe, step_ratio
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_dim,), dtype=np.float32
        )
        
        # Action: Discrete allocation for each stock (0% to 100% in bins)
        self.action_space = spaces.MultiDiscrete([num_action_bins] * self.num_stocks)
        
        print(f"Financial Environment initialized:")
        print(f"  Observation dim: {obs_dim}")
        print(f"  Action space: {self.num_stocks} stocks x {num_action_bins} bins")
        print(f"  Device: {self.device}")
    
    def _load_data_list(self) -> List[str]:
        """Load list of data file paths"""
        with open(self.data_list_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    def _load_ticker_hash(self) -> Dict:
        """Load ticker hash mapping"""
        with open(self.ticker_hash_file, 'rb') as f:
            return pickle.load(f)
    
    def _load_daily_data(self, data_idx: int) -> Dict:
        """Load data for a specific day"""
        if data_idx >= len(self.data_files):
            return None
            
        with open(self.data_files[data_idx], 'rb') as f:
            data = pickle.load(f)
        
        return {
            'features': data['trainFeature'],  # (num_stocks, feature_dim)
            'prices': data['train_in_portfolio_series'],  # (num_stocks, time_steps)
            'tickers': data['all_train_tickers']
        }
    
    def _create_observation(self, daily_data: Dict) -> np.ndarray:
        """Create observation from daily data and current state"""
        features = daily_data['features']  # (num_stocks, 249)
        tickers = daily_data['tickers']
        
        # Map tickers to unified action space
        unified_features = torch.zeros((self.num_stocks, self.feature_dim))
        current_portfolio = torch.zeros(self.num_stocks)
        
        hash_dict = self.ticker_hash['hash_D']
        for i, ticker in enumerate(tickers):
            if ticker in hash_dict:
                unified_idx = hash_dict[ticker]
                unified_features[unified_idx] = features[i]
                if self.previous_portfolio is not None:
                    current_portfolio[unified_idx] = self.previous_portfolio[unified_idx]
        
        # Aggregate features (mean pooling across stocks for now)
        aggregated_features = torch.mean(unified_features, dim=0)  # (249,)
        
        # Create full observation
        obs = torch.cat([
            aggregated_features,
            current_portfolio,
            torch.tensor([self.portfolio_value, 0.0, self.current_step / self.max_episode_steps])
        ])
        
        return obs.numpy().astype(np.float32)
    
    def _actions_to_portfolio(self, actions: np.ndarray) -> torch.Tensor:
        """Convert discrete actions to portfolio weights"""
        # Convert discrete bins to continuous weights
        weights = actions / (self.num_action_bins - 1)  # [0, 1]
        
        # Normalize to sum to 1 (portfolio constraint)
        weights = weights / (np.sum(weights) + 1e-8)
        
        return torch.from_numpy(weights).float()
    
    def _compute_returns(self, daily_data: Dict, portfolio: torch.Tensor) -> Tuple[float, float, float, float]:
        """Compute portfolio returns and Sharpe ratio"""
        prices = daily_data['prices']  # (num_stocks, time_steps)
        tickers = daily_data['tickers']
        
        # Map to unified space
        unified_prices = torch.zeros((self.num_stocks, prices.shape[1]))
        hash_dict = self.ticker_hash['hash_D']
        
        for i, ticker in enumerate(tickers):
            if ticker in hash_dict:
                unified_idx = hash_dict[ticker]
                unified_prices[unified_idx] = prices[i]
        
        # Compute returns
        if unified_prices.shape[1] > 1:
            # Portfolio shares
            initial_prices = unified_prices[:, 0] + 1e-10
            final_prices = unified_prices[:, -1]
            portfolio_shares = portfolio / initial_prices
            
            # Actual return
            actual_return = torch.sum((final_prices - initial_prices) * portfolio_shares)
            
            # Time series returns
            returns_series = torch.sum(unified_prices[:, 1:] * portfolio_shares.unsqueeze(1) - 
                                     initial_prices.unsqueeze(1) * portfolio_shares.unsqueeze(1), dim=0)
            
            mean_return = torch.mean(returns_series)
            stddev = torch.std(returns_series) + 1e-10
            sharpe = mean_return / stddev
            
            return sharpe.item(), mean_return.item(), actual_return.item(), stddev.item()
        else:
            return 0.0, 0.0, 0.0, 1.0
    
    def _compute_transaction_cost(self, new_portfolio: torch.Tensor) -> float:
        """Compute transaction costs"""
        if self.previous_portfolio is None:
            return 0.0
        
        cost = torch.sum(torch.abs(new_portfolio - self.previous_portfolio)) * self.transaction_cost
        return cost.item() * self.portfolio_value
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to start of episode"""
        super().reset(seed=seed)
        
        # Reset state
        self.current_step = 0
        self.current_data_idx = np.random.randint(0, max(1, len(self.data_files) - self.max_episode_steps))
        self.portfolio_value = 1.0
        self.previous_portfolio = torch.ones(self.num_stocks) / self.num_stocks  # Equal weights
        self.previous_prices = None
        self.episode_data = []
        
        # Load first day data
        daily_data = self._load_daily_data(self.current_data_idx)
        if daily_data is None:
            # Fallback to first file
            self.current_data_idx = 0
            daily_data = self._load_daily_data(0)
        
        observation = self._create_observation(daily_data)
        info = {'portfolio_value': self.portfolio_value, 'step': self.current_step}
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        # Load current day data
        daily_data = self._load_daily_data(self.current_data_idx)
        if daily_data is None:
            # Episode finished
            return self._create_observation({'features': torch.zeros(1, self.feature_dim), 
                                          'prices': torch.zeros(1, 1), 'tickers': []}), \
                   0.0, True, False, {'portfolio_value': self.portfolio_value}
        
        # Convert actions to portfolio
        new_portfolio = self._actions_to_portfolio(action)
        
        # Compute returns
        sharpe, mean_return, actual_return, stddev = self._compute_returns(daily_data, new_portfolio)
        
        # Update portfolio value (considering transaction costs)
        if self.current_step % self.action_update_interval == 0:
            # Actually rebalance portfolio
            transaction_cost = self._compute_transaction_cost(new_portfolio)
            self.portfolio_value = self.portfolio_value * (1 + actual_return / 100) - transaction_cost
            self.previous_portfolio = new_portfolio.clone()
        else:
            # Hold previous portfolio
            self.portfolio_value = self.portfolio_value * (1 + actual_return / 100)
        
        # Reward is Sharpe ratio
        reward = float(sharpe)
        
        # Update state
        self.current_step += 1
        self.current_data_idx += 1
        
        # Check termination
        terminated = (self.current_step >= self.max_episode_steps or 
                     self.current_data_idx >= len(self.data_files) or
                     self.portfolio_value <= 0.1)  # Prevent bankruptcy
        
        # Get next observation
        if not terminated:
            next_daily_data = self._load_daily_data(self.current_data_idx)
            if next_daily_data is None:
                terminated = True
                observation = self._create_observation({'features': torch.zeros(1, self.feature_dim), 
                                                      'prices': torch.zeros(1, 1), 'tickers': []})
            else:
                observation = self._create_observation(next_daily_data)
        else:
            observation = self._create_observation({'features': torch.zeros(1, self.feature_dim), 
                                                  'prices': torch.zeros(1, 1), 'tickers': []})
        
        info = {
            'portfolio_value': self.portfolio_value,
            'sharpe': sharpe,
            'mean_return': mean_return,
            'actual_return': actual_return,
            'stddev': stddev,
            'step': self.current_step
        }
        
        return observation, reward, terminated, False, info
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            print(f"Step: {self.current_step}, Portfolio Value: {self.portfolio_value:.4f}")
    
    def close(self):
        """Clean up environment"""
        pass


class FinancialEnvironmentWrapper:
    """Wrapper to make financial environment compatible with DQN/R2D2 interface"""
    
    def __init__(self, env: FinancialEnvironment):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        
        # For compatibility with game environment interface
        self.n_actions = np.prod(env.action_space.nvec)  # Total action combinations
        
    def reset(self):
        """Reset compatible with DQN interface"""
        obs, info = self.env.reset()
        return obs, info
    
    def step(self, action):
        """Step compatible with DQN interface"""
        # Convert single integer action to multi-discrete
        action_vector = self._decode_action(action)
        return self.env.step(action_vector)
    
    def _decode_action(self, action: int) -> np.ndarray:
        """Convert single integer action to multi-discrete action vector"""
        # Simple approach: distribute action across stocks
        num_stocks = len(self.env.action_space.nvec)
        num_bins = self.env.num_action_bins
        
        # Convert to base-n representation
        action_vector = np.zeros(num_stocks, dtype=int)
        remaining = action
        
        for i in range(num_stocks):
            action_vector[i] = remaining % num_bins
            remaining //= num_bins
            
        return action_vector
    
    def render(self, mode='human'):
        return self.env.render(mode)
    
    def close(self):
        return self.env.close()