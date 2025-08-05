import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DQN(nn.Module):
    """Deep Q-Network architecture for Atari games with optional dueling networks"""
    def __init__(self, input_shape: Tuple[int, int, int], n_actions: int, use_dueling: bool = False):
        super(DQN, self).__init__()
        c, h, w = input_shape
        self.n_actions = n_actions
        self.use_dueling = use_dueling
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate size after convolutions
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        
        # Shared feature layers
        self.fc1 = nn.Linear(linear_input_size, 512)
        
        if self.use_dueling:
            # Dueling architecture: separate value and advantage streams
            self.fc_value = nn.Linear(512, 1)  # State value V(s)
            self.fc_advantage = nn.Linear(512, n_actions)  # Action advantages A(s,a)
        else:
            # Standard DQN: direct Q-value output
            self.fc2 = nn.Linear(512, n_actions)
        
    def forward(self, x):
        x = x.float() / 255.0  # Normalize pixel values
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        
        if self.use_dueling:
            # Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
            value = self.fc_value(x)  # (batch, 1)
            advantage = self.fc_advantage(x)  # (batch, n_actions)
            
            # Combine value and advantage with mean subtraction for identifiability
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
            return q_values
        else:
            # Standard DQN
            return self.fc2(x)