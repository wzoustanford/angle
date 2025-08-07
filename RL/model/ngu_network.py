import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
from .r2d2_network import R2D2Network
from .intrinsic_rewards import NGUIntrinsicReward, Agent57IntrinsicReward
from .device_utils import get_device_manager


class NGUNetwork(R2D2Network):
    """
    Never Give Up (NGU) Network extending R2D2
    
    Combines the R2D2 architecture (CNN + LSTM + Dueling) with intrinsic motivation
    through episodic memory and Random Network Distillation.
    
    Key features:
    - Dual reward stream: extrinsic + intrinsic rewards  
    - Separate value functions for extrinsic and intrinsic returns
    - Integrated episodic memory and RND for exploration
    - Compatible with existing R2D2 training pipeline
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int], 
                 n_actions: int,
                 lstm_size: int = 512,
                 num_lstm_layers: int = 1,
                 embedding_dim: int = 128,
                 rnd_feature_dim: int = 512,
                 memory_size: int = 50000,
                 use_dual_value: bool = True):
        """
        Initialize NGU network
        
        Args:
            input_shape: Input observation shape (C, H, W)
            n_actions: Number of actions
            lstm_size: LSTM hidden size
            num_lstm_layers: Number of LSTM layers
            embedding_dim: Embedding dimension for episodic memory
            rnd_feature_dim: RND feature dimension
            memory_size: Maximum episodic memory size
            use_dual_value: Whether to use separate extrinsic/intrinsic value functions
        """
        # Initialize parent R2D2 network
        super(NGUNetwork, self).__init__(
            input_shape=input_shape,
            n_actions=n_actions,
            lstm_size=lstm_size,
            num_lstm_layers=num_lstm_layers
        )
        
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.embedding_dim = embedding_dim
        self.use_dual_value = use_dual_value
        
        # NGU Intrinsic Reward Module
        self.intrinsic_reward_module = NGUIntrinsicReward(
            input_shape=input_shape,
            embedding_dim=embedding_dim,
            rnd_feature_dim=rnd_feature_dim,
            memory_size=memory_size,
            device=get_device_manager().device
        )
        
        # Dual value functions for extrinsic and intrinsic rewards
        if use_dual_value:
            # Replace single value function with dual value functions
            self.fc_value_extrinsic = nn.Linear(lstm_size, 1)  # Extrinsic value
            self.fc_value_intrinsic = nn.Linear(lstm_size, 1)  # Intrinsic value
            
            # Keep advantage function the same (shared across reward types)
            # self.fc_advantage is inherited from R2D2Network
            
            # Remove the original value function to avoid confusion
            delattr(self, 'fc_value')
        
        # Initialize new layers
        self._init_dual_value_weights()
        
        # Training configuration
        self.intrinsic_reward_scale = 1.0
        self.extrinsic_reward_scale = 1.0
        self.intrinsic_discount = 0.99
        self.extrinsic_discount = 0.999
        
        # Episode tracking
        self.current_episode_id = 0
        self.step_count = 0
    
    def _init_dual_value_weights(self):
        """Initialize weights for dual value functions"""
        if hasattr(self, 'fc_value_extrinsic'):
            nn.init.xavier_uniform_(self.fc_value_extrinsic.weight)
            nn.init.zeros_(self.fc_value_extrinsic.bias)
        
        if hasattr(self, 'fc_value_intrinsic'):
            nn.init.xavier_uniform_(self.fc_value_intrinsic.weight)
            nn.init.zeros_(self.fc_value_intrinsic.bias)
    
    def forward(self, 
                sequences: torch.Tensor, 
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                sequence_lengths: Optional[torch.Tensor] = None,
                compute_intrinsic: bool = True,
                episode_id: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for NGU network
        
        Args:
            sequences: Input sequences (batch_size, seq_len, C, H, W)
            hidden_state: LSTM hidden state (h, c)
            sequence_lengths: Optional sequence lengths for padding
            compute_intrinsic: Whether to compute intrinsic rewards
            episode_id: Episode identifier for episodic memory
            
        Returns:
            Dictionary containing:
                - q_values_extrinsic: Q-values for extrinsic rewards
                - q_values_intrinsic: Q-values for intrinsic rewards (if dual value)
                - q_values_combined: Combined Q-values
                - hidden_state: New LSTM hidden state
                - intrinsic_rewards: Computed intrinsic rewards (if requested)
                - reward_info: Intrinsic reward breakdown
        """
        batch_size, seq_len, c, h, w = sequences.shape
        
        # Get LSTM features using parent forward (but extract components)
        # Reshape to process all frames through CNN
        flat_sequences = sequences.view(batch_size * seq_len, c, h, w)
        flat_sequences = flat_sequences.float() / 255.0
        
        # Extract CNN features 
        cnn_features = self.cnn(flat_sequences)  # (batch_size * seq_len, 512)
        lstm_input = cnn_features.view(batch_size, seq_len, self.cnn_output_size)
        
        # Initialize hidden state if needed
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size, sequences.device)
        
        # LSTM forward pass
        lstm_output, new_hidden_state = self.lstm(lstm_input, hidden_state)
        # lstm_output: (batch_size, seq_len, lstm_size)
        
        # Compute intrinsic rewards if requested
        intrinsic_rewards = None
        reward_info = None
        if compute_intrinsic:
            # Use original observations for intrinsic reward computation
            # Take last frame of each sequence element for intrinsic reward
            last_frame_sequences = sequences[:, -1]  # (batch_size, C, H, W)
            
            intrinsic_rewards, reward_info = self.intrinsic_reward_module.compute_intrinsic_reward(
                last_frame_sequences, episode_id=episode_id
            )
            # intrinsic_rewards: (batch_size,)
        
        # Compute Q-values
        if self.use_dual_value:
            # Separate value functions for extrinsic and intrinsic rewards
            values_extrinsic = self.fc_value_extrinsic(lstm_output)  # (batch, seq, 1)
            values_intrinsic = self.fc_value_intrinsic(lstm_output)   # (batch, seq, 1)
            advantages = self.fc_advantage(lstm_output)               # (batch, seq, n_actions)
            
            # Dueling architecture for both value types
            q_values_extrinsic = values_extrinsic + advantages - advantages.mean(dim=-1, keepdim=True)
            q_values_intrinsic = values_intrinsic + advantages - advantages.mean(dim=-1, keepdim=True)
            
            # Combined Q-values (weighted sum)
            q_values_combined = (self.extrinsic_reward_scale * q_values_extrinsic + 
                               self.intrinsic_reward_scale * q_values_intrinsic)
        else:
            # Single value function (like standard R2D2)
            # This is less principled but simpler
            values = self.fc_value(lstm_output)  # This would fail since we removed fc_value
            advantages = self.fc_advantage(lstm_output)
            q_values_combined = values + advantages - advantages.mean(dim=-1, keepdim=True)
            q_values_extrinsic = q_values_combined
            q_values_intrinsic = None
        
        # Prepare return dictionary
        result = {
            'q_values_combined': q_values_combined,
            'q_values_extrinsic': q_values_extrinsic,
            'hidden_state': new_hidden_state
        }
        
        if self.use_dual_value:
            result['q_values_intrinsic'] = q_values_intrinsic
        
        if intrinsic_rewards is not None:
            result['intrinsic_rewards'] = intrinsic_rewards
            result['reward_info'] = reward_info
        
        return result
    
    def forward_single_step(self, 
                           state: torch.Tensor, 
                           hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                           compute_intrinsic: bool = True,
                           episode_id: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for single timestep (action selection)
        
        Args:
            state: Single state (batch_size, C, H, W)
            hidden_state: LSTM hidden state
            compute_intrinsic: Whether to compute intrinsic rewards
            episode_id: Episode identifier
            
        Returns:
            Dictionary with Q-values and updated hidden state
        """
        # Add sequence dimension
        state_seq = state.unsqueeze(1)  # (batch_size, 1, C, H, W)
        
        # Forward through network
        result = self.forward(
            state_seq, 
            hidden_state=hidden_state,
            compute_intrinsic=compute_intrinsic,
            episode_id=episode_id
        )
        
        # Remove sequence dimension from Q-values
        for key in ['q_values_combined', 'q_values_extrinsic', 'q_values_intrinsic']:
            if key in result and result[key] is not None:
                result[key] = result[key].squeeze(1)  # (batch_size, n_actions)
        
        return result
    
    def update_intrinsic_networks(self, observations: torch.Tensor) -> Dict[str, float]:
        """
        Update intrinsic reward networks (RND predictor, etc.)
        
        Args:
            observations: Batch of observations (batch_size, C, H, W)
            
        Returns:
            Dictionary of losses
        """
        return self.intrinsic_reward_module.update_networks(observations)
    
    def reset_episode(self, episode_id: Optional[str] = None):
        """Reset episode-specific state"""
        self.current_episode_id += 1
        if episode_id is None:
            episode_id = f"episode_{self.current_episode_id}"
        
        self.intrinsic_reward_module.reset_episode(episode_id)
    
    def get_intrinsic_statistics(self) -> Dict:
        """Get intrinsic reward statistics"""
        return self.intrinsic_reward_module.get_statistics()
    
    def save_ngu_state(self) -> Dict:
        """Save NGU-specific state"""
        return {
            'network_state': self.state_dict(),
            'intrinsic_module_state': self.intrinsic_reward_module.save_state(),
            'current_episode_id': self.current_episode_id,
            'step_count': self.step_count,
            'config': {
                'use_dual_value': self.use_dual_value,
                'intrinsic_reward_scale': self.intrinsic_reward_scale,
                'extrinsic_reward_scale': self.extrinsic_reward_scale,
                'intrinsic_discount': self.intrinsic_discount,
                'extrinsic_discount': self.extrinsic_discount,
            }
        }
    
    def load_ngu_state(self, state: Dict):
        """Load NGU-specific state"""
        self.load_state_dict(state['network_state'])
        self.intrinsic_reward_module.load_state(state['intrinsic_module_state'])
        
        self.current_episode_id = state.get('current_episode_id', 0)
        self.step_count = state.get('step_count', 0)
        
        if 'config' in state:
            config = state['config']
            self.intrinsic_reward_scale = config.get('intrinsic_reward_scale', 1.0)
            self.extrinsic_reward_scale = config.get('extrinsic_reward_scale', 1.0)
            self.intrinsic_discount = config.get('intrinsic_discount', 0.99)
            self.extrinsic_discount = config.get('extrinsic_discount', 0.999)


class Agent57Network(NGUNetwork):
    """
    Agent57 Network extending NGU with meta-learning capabilities
    
    Adds multiple recurrent networks (LSTMs) for different exploration strategies
    and a meta-controller for policy selection.
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int], 
                 n_actions: int,
                 num_policies: int = 32,
                 lstm_size: int = 512,
                 num_lstm_layers: int = 1,
                 embedding_dim: int = 128,
                 rnd_feature_dim: int = 512,
                 memory_size: int = 50000):
        """
        Initialize Agent57 network
        
        Args:
            input_shape: Input observation shape (C, H, W)
            n_actions: Number of actions
            num_policies: Number of meta-policies
            lstm_size: LSTM hidden size
            num_lstm_layers: Number of LSTM layers per policy
            embedding_dim: Embedding dimension
            rnd_feature_dim: RND feature dimension
            memory_size: Episodic memory size per policy
        """
        # Initialize base NGU network
        super(Agent57Network, self).__init__(
            input_shape=input_shape,
            n_actions=n_actions,
            lstm_size=lstm_size,
            num_lstm_layers=num_lstm_layers,
            embedding_dim=embedding_dim,
            rnd_feature_dim=rnd_feature_dim,
            memory_size=memory_size,
            use_dual_value=True
        )
        
        self.num_policies = num_policies
        
        # Replace single LSTM with multiple LSTMs (one per policy)
        delattr(self, 'lstm')
        
        self.policy_lstms = nn.ModuleList([
            nn.LSTM(
                input_size=self.cnn_output_size,
                hidden_size=lstm_size,
                num_layers=num_lstm_layers,
                batch_first=True
            ) for _ in range(num_policies)
        ])
        
        # Replace NGU intrinsic reward with Agent57 version
        self.intrinsic_reward_module = Agent57IntrinsicReward(
            input_shape=input_shape,
            num_policies=num_policies,
            embedding_dim=embedding_dim,
            rnd_feature_dim=rnd_feature_dim,
            memory_size=memory_size,
            device=get_device_manager().device
        )
        
        # Meta-controller for policy selection (simple linear network)
        self.meta_controller = nn.Sequential(
            nn.Linear(lstm_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_policies),
            nn.Softmax(dim=-1)
        )
        
        # Current active policy
        self.current_policy_id = 0
        
        # Initialize new components
        self._init_agent57_weights()
    
    def _init_agent57_weights(self):
        """Initialize Agent57-specific weights"""
        for m in self.meta_controller.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def init_hidden(self, batch_size: int, device: torch.device, policy_id: Optional[int] = None):
        """Initialize hidden state for specific policy"""
        if policy_id is None:
            policy_id = self.current_policy_id
        
        lstm = self.policy_lstms[policy_id]
        h_0 = torch.zeros(lstm.num_layers, batch_size, lstm.hidden_size, device=device)
        c_0 = torch.zeros(lstm.num_layers, batch_size, lstm.hidden_size, device=device)
        return (h_0, c_0)
    
    def set_policy(self, policy_id: int):
        """Set the active policy"""
        assert 0 <= policy_id < self.num_policies
        self.current_policy_id = policy_id
        self.intrinsic_reward_module.set_active_policy(policy_id)
    
    def select_policy(self, lstm_features: torch.Tensor) -> int:
        """Use meta-controller to select policy (for advanced Agent57)"""
        with torch.no_grad():
            # Take mean across sequence dimension
            mean_features = lstm_features.mean(dim=1)  # (batch_size, lstm_size)
            
            # Get policy probabilities
            policy_probs = self.meta_controller(mean_features)  # (batch_size, num_policies)
            
            # Sample policy (or take argmax for deterministic)
            policy_id = torch.multinomial(policy_probs[0], 1).item()
            
        return policy_id
    
    def forward(self, 
                sequences: torch.Tensor, 
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                sequence_lengths: Optional[torch.Tensor] = None,
                policy_id: Optional[int] = None,
                compute_intrinsic: bool = True,
                episode_id: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Agent57 network
        
        Args:
            sequences: Input sequences (batch_size, seq_len, C, H, W)
            hidden_state: LSTM hidden state for specific policy
            sequence_lengths: Optional sequence lengths
            policy_id: Policy ID to use (uses current if None)
            compute_intrinsic: Whether to compute intrinsic rewards
            episode_id: Episode identifier
            
        Returns:
            Dictionary with Q-values, hidden state, and intrinsic rewards
        """
        if policy_id is None:
            policy_id = self.current_policy_id
        
        batch_size, seq_len, c, h, w = sequences.shape
        
        # CNN feature extraction (shared across policies)
        flat_sequences = sequences.view(batch_size * seq_len, c, h, w)
        flat_sequences = flat_sequences.float() / 255.0
        cnn_features = self.cnn(flat_sequences)
        lstm_input = cnn_features.view(batch_size, seq_len, self.cnn_output_size)
        
        # Policy-specific LSTM
        lstm = self.policy_lstms[policy_id]
        
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size, sequences.device, policy_id)
        
        lstm_output, new_hidden_state = lstm(lstm_input, hidden_state)
        
        # Compute intrinsic rewards using policy-specific module
        intrinsic_rewards = None
        reward_info = None
        if compute_intrinsic:
            last_frame_sequences = sequences[:, -1]  # (batch_size, C, H, W)
            intrinsic_rewards, reward_info = self.intrinsic_reward_module.compute_policy_intrinsic_reward(
                last_frame_sequences, policy_id=policy_id, episode_id=episode_id
            )
        
        # Value functions and advantages (shared across policies)
        values_extrinsic = self.fc_value_extrinsic(lstm_output)
        values_intrinsic = self.fc_value_intrinsic(lstm_output)
        advantages = self.fc_advantage(lstm_output)
        
        # Dueling Q-values
        q_values_extrinsic = values_extrinsic + advantages - advantages.mean(dim=-1, keepdim=True)
        q_values_intrinsic = values_intrinsic + advantages - advantages.mean(dim=-1, keepdim=True)
        q_values_combined = (self.extrinsic_reward_scale * q_values_extrinsic + 
                           self.intrinsic_reward_scale * q_values_intrinsic)
        
        # Prepare result
        result = {
            'q_values_combined': q_values_combined,
            'q_values_extrinsic': q_values_extrinsic,
            'q_values_intrinsic': q_values_intrinsic,
            'hidden_state': new_hidden_state,
            'policy_id': policy_id
        }
        
        if intrinsic_rewards is not None:
            result['intrinsic_rewards'] = intrinsic_rewards
            result['reward_info'] = reward_info
        
        return result
    
    def get_agent57_statistics(self) -> Dict:
        """Get comprehensive Agent57 statistics"""
        base_stats = self.get_intrinsic_statistics()
        policy_stats = self.intrinsic_reward_module.get_policy_statistics()
        
        return {**base_stats, **policy_stats}


def test_ngu_network():
    """Test function for NGU network"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing NGU Network on device: {device}")
    
    # Test parameters
    batch_size = 2
    seq_len = 4
    input_shape = (4, 84, 84)  # Atari frame stack
    n_actions = 6
    
    # Create NGU network
    ngu_net = NGUNetwork(
        input_shape=input_shape,
        n_actions=n_actions,
        lstm_size=128,  # Smaller for testing
        embedding_dim=32,  # Smaller for testing
        rnd_feature_dim=64,  # Smaller for testing
        memory_size=100,  # Much smaller for testing
        use_dual_value=True
    ).to(device)
    
    print(f"Created NGU network with {sum(p.numel() for p in ngu_net.parameters())} parameters")
    
    # Test input
    sequences = torch.randint(0, 256, (batch_size, seq_len, *input_shape), 
                             dtype=torch.float32, device=device)
    
    print(f"Input sequences shape: {sequences.shape}")
    
    # Forward pass
    result = ngu_net.forward(sequences, compute_intrinsic=True, episode_id="test_episode")
    
    print(f"Q-values combined shape: {result['q_values_combined'].shape}")
    print(f"Q-values extrinsic shape: {result['q_values_extrinsic'].shape}")
    print(f"Q-values intrinsic shape: {result['q_values_intrinsic'].shape}")
    print(f"Intrinsic rewards: {result['intrinsic_rewards']}")
    
    # Test single step
    single_state = sequences[:, 0]  # (batch_size, C, H, W)
    single_result = ngu_net.forward_single_step(single_state, compute_intrinsic=True)
    print(f"Single step Q-values shape: {single_result['q_values_combined'].shape}")
    
    # Test network updates
    observations = sequences[:, -1]  # Last frame
    losses = ngu_net.update_intrinsic_networks(observations)
    print(f"Intrinsic network losses: {losses}")
    
    # Test Agent57 network
    print("\n--- Testing Agent57 Network ---")
    agent57_net = Agent57Network(
        input_shape=input_shape,
        n_actions=n_actions,
        num_policies=4,  # Smaller for testing
        lstm_size=128,
        embedding_dim=32,
        rnd_feature_dim=64,
        memory_size=100
    ).to(device)
    
    print(f"Created Agent57 network with {sum(p.numel() for p in agent57_net.parameters())} parameters")
    
    # Test different policies
    for policy_id in [0, 2]:
        agent57_net.set_policy(policy_id)
        result57 = agent57_net.forward(sequences, compute_intrinsic=True, episode_id="test_episode")
        print(f"Policy {policy_id}: Q-values shape {result57['q_values_combined'].shape}, "
              f"intrinsic reward mean: {result57['intrinsic_rewards'].mean():.4f}")
    
    print("✓ NGU Network test passed!")


if __name__ == "__main__":
    test_ngu_network()