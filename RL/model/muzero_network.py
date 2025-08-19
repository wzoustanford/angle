import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


def support_to_scalar(logits: torch.Tensor, support_size: int) -> torch.Tensor:
    """
    Transform categorical representation to scalar
    """
    probabilities = torch.softmax(logits, dim=-1)
    support = torch.arange(-support_size, support_size + 1, dtype=torch.float32, device=logits.device)
    support = support.expand_as(probabilities)
    return torch.sum(support * probabilities, dim=-1)


def scalar_to_support(scalar: torch.Tensor, support_size: int) -> torch.Tensor:
    """
    Transform scalar to categorical representation
    """
    scalar = torch.clamp(scalar, -support_size, support_size)
    floor = scalar.floor()
    prob = scalar - floor
    
    logits = torch.zeros(scalar.shape[0], 2 * support_size + 1, device=scalar.device)
    
    lower_index = (floor + support_size).long()
    upper_index = (floor + support_size + 1).long()
    
    # Handle edge cases
    lower_index = torch.clamp(lower_index, 0, 2 * support_size)
    upper_index = torch.clamp(upper_index, 0, 2 * support_size)
    
    # Distribute probability
    logits.scatter_add_(1, lower_index.unsqueeze(1), (1 - prob).unsqueeze(1))
    logits.scatter_add_(1, upper_index.unsqueeze(1), prob.unsqueeze(1))
    
    return logits


class ResidualBlock(nn.Module):
    """Residual block for deeper networks"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class RepresentationNetwork(nn.Module):
    """
    Representation function h: o -> s
    Encodes observations into latent state representation
    """
    
    def __init__(self, observation_shape: Tuple[int, int, int], 
                 num_channels: int = 256, num_blocks: int = 2):
        super().__init__()
        c, h, w = observation_shape
        
        # Initial convolution to increase channels
        self.conv = nn.Conv2d(c, num_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(num_channels)
        
        # Downsample to smaller spatial dimensions
        self.downsample = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_blocks)
        ])
        
        # Output dimensions after downsampling
        self.out_h = h // 4
        self.out_w = w // 4
        self.out_channels = num_channels
        
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observation: (batch, channels, height, width)
        Returns:
            hidden_state: (batch, channels, height//4, width//4)
        """
        x = F.relu(self.bn(self.conv(observation)))
        x = self.downsample(x)
        
        for block in self.blocks:
            x = block(x)
            
        return x


class DynamicsNetwork(nn.Module):
    """
    Dynamics function g: (s, a) -> (r, s')
    Predicts next state and reward given current state and action
    """
    
    def __init__(self, state_channels: int, state_h: int, state_w: int,
                 action_space_size: int, num_channels: int = 256,
                 support_size: int = 300, num_blocks: int = 2):
        super().__init__()
        
        self.action_space_size = action_space_size
        self.state_channels = state_channels
        self.state_h = state_h
        self.state_w = state_w
        
        # Encode action as spatial planes (following MuZero paper)
        self.conv = nn.Conv2d(state_channels + action_space_size, num_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(num_channels)
        
        # Residual blocks for processing
        self.blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_blocks)
        ])
        
        # Predict next state
        self.state_head = nn.Conv2d(num_channels, state_channels, 3, padding=1)
        
        # Predict reward (categorical representation)
        self.reward_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, 1),
            nn.Flatten(),
            nn.Linear(state_h * state_w, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * support_size + 1)
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (batch, channels, height, width)
            action: (batch,) action indices
        Returns:
            reward: (batch, 2*support_size+1) categorical reward
            next_state: (batch, channels, height, width)
        """
        batch_size = state.shape[0]
        
        # Create action planes
        action_planes = torch.zeros(
            batch_size, self.action_space_size, self.state_h, self.state_w,
            device=state.device
        )
        action_planes[torch.arange(batch_size), action] = 1.0
        
        # Concatenate state and action
        x = torch.cat([state, action_planes], dim=1)
        
        # Process through network
        x = F.relu(self.bn(self.conv(x)))
        
        for block in self.blocks:
            x = block(x)
        
        # Predict next state and reward
        next_state = self.state_head(x)
        reward = self.reward_head(x)
        
        return reward, next_state


class PredictionNetwork(nn.Module):
    """
    Prediction function f: s -> (p, v)
    Predicts policy and value from state representation
    """
    
    def __init__(self, state_channels: int, state_h: int, state_w: int,
                 action_space_size: int, num_channels: int = 256,
                 support_size: int = 300, num_blocks: int = 2):
        super().__init__()
        
        # Initial processing
        self.conv = nn.Conv2d(state_channels, num_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(num_channels)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_blocks)
        ])
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * state_h * state_w, action_space_size)
        )
        
        # Value head (categorical representation)
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(state_h * state_w, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * support_size + 1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (batch, channels, height, width)
        Returns:
            policy_logits: (batch, action_space_size)
            value: (batch, 2*support_size+1) categorical value
        """
        x = F.relu(self.bn(self.conv(state)))
        
        for block in self.blocks:
            x = block(x)
        
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return policy_logits, value


class MuZeroNetwork(nn.Module):
    """
    Complete MuZero network combining representation, dynamics, and prediction functions
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.action_space_size = config.action_space_size
        self.support_size = config.support_size
        
        # Calculate state dimensions after representation network
        _, h, w = config.observation_shape
        state_h = h // 4  # After 2 stride-2 convolutions
        state_w = w // 4
        
        # Initialize three networks
        self.representation = RepresentationNetwork(
            config.observation_shape,
            config.hidden_size
        )
        
        self.dynamics = DynamicsNetwork(
            config.hidden_size,
            state_h,
            state_w,
            config.action_space_size,
            config.hidden_size,
            config.support_size
        )
        
        self.prediction = PredictionNetwork(
            config.hidden_size,
            state_h,
            state_w,
            config.action_space_size,
            config.hidden_size,
            config.support_size
        )
        
    def initial_inference(self, observation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Initial inference for root node
        h(o) -> s; f(s) -> p, v
        """
        state = self.representation(observation)
        policy_logits, value_logits = self.prediction(state)
        
        # Convert categorical to scalar if needed
        value = support_to_scalar(value_logits, self.support_size)
        
        return {
            'state': state,
            'policy_logits': policy_logits,
            'value': value,
            'value_logits': value_logits
        }
    
    def recurrent_inference(self, state: torch.Tensor, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Recurrent inference for tree search
        g(s, a) -> r, s'; f(s') -> p, v
        """
        reward_logits, next_state = self.dynamics(state, action)
        policy_logits, value_logits = self.prediction(next_state)
        
        # Convert categorical to scalar if needed
        reward = support_to_scalar(reward_logits, self.support_size)
        value = support_to_scalar(value_logits, self.support_size)
        
        return {
            'state': next_state,
            'reward': reward,
            'reward_logits': reward_logits,
            'policy_logits': policy_logits,
            'value': value,
            'value_logits': value_logits
        }
    
    def forward(self, observations: torch.Tensor, 
                actions: Optional[List[torch.Tensor]] = None) -> Dict[str, List[torch.Tensor]]:
        """
        Forward pass for training
        Unrolls the model for multiple steps
        """
        outputs = {
            'policy': [],
            'value': [],
            'reward': [],
            'value_logits': [],
            'reward_logits': [],
            'policy_logits': []
        }
        
        # Initial inference
        initial_output = self.initial_inference(observations)
        outputs['policy'].append(F.softmax(initial_output['policy_logits'], dim=-1))
        outputs['value'].append(initial_output['value'])
        outputs['reward'].append(torch.zeros_like(initial_output['value']))
        outputs['value_logits'].append(initial_output['value_logits'])
        outputs['policy_logits'].append(initial_output['policy_logits'])
        outputs['reward_logits'].append(
            scalar_to_support(torch.zeros_like(initial_output['value']), self.support_size)
        )
        
        # Recurrent inference for unroll steps
        if actions is not None:
            state = initial_output['state']
            for action in actions:
                output = self.recurrent_inference(state, action)
                outputs['policy'].append(F.softmax(output['policy_logits'], dim=-1))
                outputs['value'].append(output['value'])
                outputs['reward'].append(output['reward'])
                outputs['value_logits'].append(output['value_logits'])
                outputs['reward_logits'].append(output['reward_logits'])
                outputs['policy_logits'].append(output['policy_logits'])
                state = output['state']
        
        return outputs