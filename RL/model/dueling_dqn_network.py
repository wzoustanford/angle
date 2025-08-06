import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DuelingDQN(nn.Module):
    """Dueling Deep Q-Network architecture for Atari games
    
    Implements the dueling architecture from "Dueling Network Architectures 
    for Deep Reinforcement Learning" (ICML 2016) by Wang et al.
    
    Architecture:
    - Shared CNN feature extraction
    - Separate value stream V(s) and advantage stream A(s,a)
    - Combined as Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], n_actions: int):
        super(DuelingDQN, self).__init__()
        c, h, w = input_shape
        
        # Shared convolutional layers (same as standard DQN)
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate size after convolutions
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        
        # Shared fully connected layer
        self.fc_shared = nn.Linear(linear_input_size, 512)
        
        # Dueling streams
        self.fc_value = nn.Linear(512, 1)        # State value V(s)
        self.fc_advantage = nn.Linear(512, n_actions)  # Action advantages A(s,a)
        
    def forward(self, x):
        x = x.float() / 255.0  # Normalize pixel values
        
        # Shared CNN feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_shared(x))
        
        # Dueling streams
        value = self.fc_value(x)        # (batch, 1)
        advantage = self.fc_advantage(x) # (batch, n_actions)
        
        # Combine streams: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        # Subtracting mean advantage makes the representation unique
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values