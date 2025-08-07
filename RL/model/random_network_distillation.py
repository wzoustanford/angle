import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
from .device_utils import get_device_manager


class RNDTarget(nn.Module):
    """
    Random Network Distillation Target Network
    
    This is a randomly initialized network that remains fixed throughout training.
    It generates "random" but consistent representations of input states.
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int],
                 feature_dim: int = 512):
        """
        Initialize RND target network
        
        Args:
            input_shape: Input observation shape (C, H, W)
            feature_dim: Output feature dimension
        """
        super(RNDTarget, self).__init__()
        
        self.input_shape = input_shape
        self.feature_dim = feature_dim
        
        # CNN layers similar to DQN but potentially different architecture
        # Using different architecture to ensure target and predictor are different
        c, h, w = input_shape
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate conv output size
        self.conv_output_size = self._calc_conv_output_size(input_shape)
        
        # Final layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )
        
        # Initialize with random weights and freeze
        self._init_random_weights()
        self._freeze_network()
    
    def _calc_conv_output_size(self, input_shape: Tuple[int, int, int]):
        """Calculate the output size after convolutions"""
        c, h, w = input_shape
        
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        # Apply conv layers: 8x4, 4x2, 3x1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        return convw * convh * 64
    
    def _init_random_weights(self):
        """Initialize with random weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _freeze_network(self):
        """Freeze all parameters - target network never updates"""
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through target network
        
        Args:
            observations: Input observations (batch_size, C, H, W)
            
        Returns:
            features: Random target features (batch_size, feature_dim)
        """
        # Normalize input
        if observations.dtype != torch.float32:
            observations = observations.float()
        if observations.max() > 1.0:
            observations = observations / 255.0
        
        # Forward pass
        conv_features = self.conv_layers(observations)
        features = self.fc_layers(conv_features)
        
        return features


class RNDPredictor(nn.Module):
    """
    Random Network Distillation Predictor Network
    
    This network is trained to predict the output of the target network.
    The prediction error serves as an intrinsic reward signal.
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int],
                 feature_dim: int = 512):
        """
        Initialize RND predictor network
        
        Args:
            input_shape: Input observation shape (C, H, W) 
            feature_dim: Output feature dimension (should match target)
        """
        super(RNDPredictor, self).__init__()
        
        self.input_shape = input_shape
        self.feature_dim = feature_dim
        
        # CNN layers - can be similar to target but with different initialization
        c, h, w = input_shape
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate conv output size
        self.conv_output_size = self._calc_conv_output_size(input_shape)
        
        # Final layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )
        
        # Initialize with Xavier initialization
        self._init_weights()
    
    def _calc_conv_output_size(self, input_shape: Tuple[int, int, int]):
        """Calculate the output size after convolutions"""
        c, h, w = input_shape
        
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        # Apply conv layers: 8x4, 4x2, 3x1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        return convw * convh * 64
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through predictor network
        
        Args:
            observations: Input observations (batch_size, C, H, W)
            
        Returns:
            predictions: Predicted features (batch_size, feature_dim)
        """
        # Normalize input
        if observations.dtype != torch.float32:
            observations = observations.float()
        if observations.max() > 1.0:
            observations = observations / 255.0
        
        # Forward pass
        conv_features = self.conv_layers(observations)
        predictions = self.fc_layers(conv_features)
        
        return predictions


class RNDModule:
    """
    Complete Random Network Distillation Module
    
    Combines target and predictor networks with intrinsic reward computation.
    Used as a component in NGU for curiosity-driven exploration.
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int],
                 feature_dim: int = 512,
                 learning_rate: float = 1e-4,
                 device: torch.device = None):
        """
        Initialize RND module
        
        Args:
            input_shape: Input observation shape (C, H, W)
            feature_dim: Feature dimension for both networks
            learning_rate: Learning rate for predictor network
            device: Device to run on
        """
        self.input_shape = input_shape
        self.feature_dim = feature_dim
        self.device = device or get_device_manager().get_device()
        
        # Initialize networks
        self.target_network = RNDTarget(input_shape, feature_dim).to(self.device)
        self.predictor_network = RNDPredictor(input_shape, feature_dim).to(self.device)
        
        # Optimizer for predictor network only
        self.optimizer = torch.optim.Adam(
            self.predictor_network.parameters(), 
            lr=learning_rate
        )
        
        # Running statistics for normalization
        self.prediction_error_mean = 0.0
        self.prediction_error_std = 1.0
        self.error_update_count = 0
        self.error_momentum = 0.99
        
        # Statistics tracking
        self.total_updates = 0
        self.recent_errors = []
        self.max_recent_errors = 1000
    
    def compute_intrinsic_reward(self, observations: torch.Tensor, 
                                normalize: bool = True) -> torch.Tensor:
        """
        Compute intrinsic reward based on prediction error
        
        Args:
            observations: Input observations (batch_size, C, H, W)
            normalize: Whether to normalize rewards
            
        Returns:
            intrinsic_rewards: Intrinsic rewards (batch_size,)
        """
        with torch.no_grad():
            # Get target and predicted features
            target_features = self.target_network(observations)
            predicted_features = self.predictor_network(observations)
            
            # Compute MSE prediction error
            prediction_errors = F.mse_loss(
                predicted_features, target_features, reduction='none'
            ).mean(dim=1)  # (batch_size,)
            
            # Update running statistics
            if normalize and self.error_update_count > 0:
                # Normalize using running statistics
                normalized_errors = (prediction_errors - self.prediction_error_mean) / (
                    self.prediction_error_std + 1e-8
                )
                intrinsic_rewards = torch.clamp(normalized_errors, 0.0, 5.0)
            else:
                # Use raw prediction errors
                intrinsic_rewards = prediction_errors
            
            # Update running statistics
            self._update_error_stats(prediction_errors)
            
            return intrinsic_rewards
    
    def update_predictor(self, observations: torch.Tensor) -> float:
        """
        Update the predictor network to minimize prediction error
        
        Args:
            observations: Input observations (batch_size, C, H, W)
            
        Returns:
            loss: Prediction loss
        """
        # Get target features (no gradients)
        with torch.no_grad():
            target_features = self.target_network(observations)
        
        # Get predicted features
        predicted_features = self.predictor_network(observations)
        
        # Compute loss
        loss = F.mse_loss(predicted_features, target_features)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.predictor_network.parameters(), 5.0)
        
        self.optimizer.step()
        
        self.total_updates += 1
        
        return loss.item()
    
    def _update_error_stats(self, errors: torch.Tensor):
        """Update running statistics for prediction errors"""
        errors_np = errors.detach().cpu().numpy()
        
        # Update recent errors
        self.recent_errors.extend(errors_np.tolist())
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors = self.recent_errors[-self.max_recent_errors:]
        
        # Update running mean and std
        batch_mean = np.mean(errors_np)
        
        if self.error_update_count == 0:
            self.prediction_error_mean = batch_mean
            self.prediction_error_std = max(np.std(errors_np), 1e-8)
        else:
            # Exponential moving average
            self.prediction_error_mean = (
                self.error_momentum * self.prediction_error_mean + 
                (1 - self.error_momentum) * batch_mean
            )
            
            if len(self.recent_errors) > 10:
                self.prediction_error_std = max(np.std(self.recent_errors), 1e-8)
        
        self.error_update_count += 1
    
    def get_stats(self) -> Dict:
        """Get RND statistics for monitoring"""
        return {
            'total_updates': self.total_updates,
            'prediction_error_mean': self.prediction_error_mean,
            'prediction_error_std': self.prediction_error_std,
            'recent_error_count': len(self.recent_errors),
            'recent_error_mean': np.mean(self.recent_errors) if self.recent_errors else 0.0,
            'recent_error_std': np.std(self.recent_errors) if self.recent_errors else 0.0
        }
    
    def save_state(self) -> Dict:
        """Save RND state for checkpointing"""
        return {
            'predictor_state_dict': self.predictor_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'prediction_error_mean': self.prediction_error_mean,
            'prediction_error_std': self.prediction_error_std,
            'error_update_count': self.error_update_count,
            'total_updates': self.total_updates,
            'recent_errors': self.recent_errors[-100:]  # Save last 100 errors
        }
    
    def load_state(self, state: Dict):
        """Load RND state from checkpoint"""
        self.predictor_network.load_state_dict(state['predictor_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.prediction_error_mean = state.get('prediction_error_mean', 0.0)
        self.prediction_error_std = state.get('prediction_error_std', 1.0)
        self.error_update_count = state.get('error_update_count', 0)
        self.total_updates = state.get('total_updates', 0)
        self.recent_errors = state.get('recent_errors', [])


def test_rnd_module():
    """Test function for RND module"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing RND on device: {device}")
    
    # Test parameters
    batch_size = 8
    input_shape = (4, 84, 84)  # Atari frame stack
    feature_dim = 512
    
    # Create RND module
    rnd = RNDModule(
        input_shape=input_shape,
        feature_dim=feature_dim,
        device=device
    )
    
    # Generate test observations
    observations = torch.randint(0, 256, (batch_size, *input_shape), 
                                dtype=torch.float32, device=device)
    
    print(f"Input shape: {observations.shape}")
    
    # Test intrinsic reward computation
    intrinsic_rewards = rnd.compute_intrinsic_reward(observations)
    print(f"Intrinsic rewards shape: {intrinsic_rewards.shape}")
    print(f"Intrinsic rewards: {intrinsic_rewards}")
    
    # Test predictor update
    initial_loss = rnd.update_predictor(observations)
    print(f"Initial prediction loss: {initial_loss:.4f}")
    
    # Train for a few steps
    for step in range(10):
        loss = rnd.update_predictor(observations)
        if step % 5 == 0:
            print(f"Step {step}: Loss = {loss:.4f}")
    
    # Test stats
    stats = rnd.get_stats()
    print("RND Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test with different observations
    new_observations = torch.randint(0, 256, (batch_size, *input_shape), 
                                    dtype=torch.float32, device=device)
    new_rewards = rnd.compute_intrinsic_reward(new_observations)
    print(f"New intrinsic rewards: {new_rewards}")
    
    print("✓ RND module test passed!")


if __name__ == "__main__":
    test_rnd_module()