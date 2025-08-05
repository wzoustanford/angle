import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .dqn_network import DQN
from .device_utils import get_device_manager


class R2D2Network(nn.Module):
    """
    R2D2 Network: CNN + LSTM for sequential decision making
    
    Architecture:
    - Reuses existing DQN CNN layers for feature extraction
    - Adds LSTM for temporal modeling across sequences
    - Maintains compatibility with existing DQN for easy switching
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], n_actions: int, 
                 lstm_size: int = 512, num_lstm_layers: int = 1):
        super(R2D2Network, self).__init__()
        
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.lstm_size = lstm_size
        self.num_lstm_layers = num_lstm_layers
        
        # Reuse existing DQN CNN architecture
        self.cnn = self._build_cnn(input_shape)
        
        # Calculate CNN output size
        self.cnn_output_size = self._get_cnn_output_size(input_shape)
        
        # LSTM layer for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=lstm_size,
            num_layers=num_lstm_layers,
            batch_first=True  # (batch, seq, features)
        )
        
        # Output layers
        self.fc_value = nn.Linear(lstm_size, 1)  # State value
        self.fc_advantage = nn.Linear(lstm_size, n_actions)  # Action advantages
        
    def _build_cnn(self, input_shape: Tuple[int, int, int]):
        """Build CNN layers identical to DQN for compatibility"""
        c, h, w = input_shape
        
        return nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self._calc_conv_output_size(input_shape), 512),
            nn.ReLU()
        )
    
    def _calc_conv_output_size(self, input_shape: Tuple[int, int, int]):
        """Calculate the output size after convolutions"""
        c, h, w = input_shape
        
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        # Apply conv layers: 8x4, 4x2, 3x1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        return convw * convh * 64
    
    def _get_cnn_output_size(self, input_shape: Tuple[int, int, int]):
        """Get the output size of CNN (should be 512 for compatibility)"""
        return 512
    
    def init_hidden(self, batch_size: int, device: torch.device):
        """Initialize LSTM hidden state"""
        h_0 = torch.zeros(self.num_lstm_layers, batch_size, self.lstm_size, device=device)
        c_0 = torch.zeros(self.num_lstm_layers, batch_size, self.lstm_size, device=device)
        return (h_0, c_0)
    
    def forward(self, sequences: torch.Tensor, 
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                sequence_lengths: Optional[torch.Tensor] = None):
        """
        Forward pass for R2D2
        
        Args:
            sequences: (batch_size, seq_len, channels, height, width)
            hidden_state: Optional LSTM hidden state (h, c)
            sequence_lengths: Optional tensor of actual sequence lengths for padding
            
        Returns:
            q_values: (batch_size, seq_len, n_actions)
            new_hidden_state: Updated LSTM hidden state
        """
        batch_size, seq_len, c, h, w = sequences.shape
        
        # Reshape to process all frames through CNN
        # (batch_size * seq_len, channels, height, width)
        flat_sequences = sequences.view(batch_size * seq_len, c, h, w)
        
        # Extract features using CNN
        # Normalize pixel values like in DQN
        flat_sequences = flat_sequences.float() / 255.0
        cnn_features = self.cnn(flat_sequences)  # (batch_size * seq_len, 512)
        
        # Reshape back to sequences: (batch_size, seq_len, 512)
        lstm_input = cnn_features.view(batch_size, seq_len, self.cnn_output_size)
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size, sequences.device)
        
        # LSTM forward pass
        lstm_output, new_hidden_state = self.lstm(lstm_input, hidden_state)
        
        # Dueling DQN: separate value and advantage streams
        values = self.fc_value(lstm_output)  # (batch, seq, 1)
        advantages = self.fc_advantage(lstm_output)  # (batch, seq, n_actions)
        
        # Combine value and advantage: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = values + advantages - advantages.mean(dim=-1, keepdim=True)
        
        return q_values, new_hidden_state
    
    def forward_single_step(self, state: torch.Tensor, 
                           hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Forward pass for single timestep (useful for action selection)
        
        Args:
            state: (batch_size, channels, height, width) - single frame stack
            hidden_state: LSTM hidden state
            
        Returns:
            q_values: (batch_size, n_actions)
            new_hidden_state: Updated LSTM hidden state
        """
        # Add sequence dimension: (batch_size, 1, channels, height, width)
        state_seq = state.unsqueeze(1)
        
        # Forward through network
        q_values_seq, new_hidden_state = self.forward(state_seq, hidden_state)
        
        # Remove sequence dimension: (batch_size, n_actions)
        q_values = q_values_seq.squeeze(1)
        
        return q_values, new_hidden_state


class R2D2CompatibleDQN(DQN):
    """
    DQN wrapper that can be used interchangeably with R2D2Network
    Useful for comparison and gradual migration
    """
    
    def forward_single_step(self, state: torch.Tensor, hidden_state=None):
        """Compatibility method for single step forward"""
        q_values = self.forward(state)
        # Return None hidden state for compatibility
        return q_values, None
    
    def init_hidden(self, batch_size: int, device: torch.device):
        """Compatibility method - returns None since DQN has no hidden state"""
        return None