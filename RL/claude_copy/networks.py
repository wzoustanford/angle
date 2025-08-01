# networks.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration"""
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

class AtariCNN(nn.Module):
    """Basic CNN for Atari games"""
    def __init__(self, num_inputs, hidden_size=512):
        super().__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        
        # Compute conv output size
        self.conv_output_size = 7 * 7 * 64
        self.fc = nn.Linear(self.conv_output_size, hidden_size)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

class DQN(nn.Module):
    """Deep Q-Network"""
    def __init__(self, num_inputs, num_actions, dueling=False, noisy=False, sigma_init=0.5):
        super().__init__()
        self.num_actions = num_actions
        self.dueling = dueling
        self.noisy = noisy
        
        self.features = AtariCNN(num_inputs)
        
        if self.dueling:
            # Value stream
            if noisy:
                self.value_stream = nn.Sequential(
                    NoisyLinear(512, 512, sigma_init),
                    nn.ReLU(),
                    NoisyLinear(512, 1, sigma_init)
                )
                # Advantage stream
                self.advantage_stream = nn.Sequential(
                    NoisyLinear(512, 512, sigma_init),
                    nn.ReLU(),
                    NoisyLinear(512, num_actions, sigma_init)
                )
            else:
                self.value_stream = nn.Sequential(
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 1)
                )
                self.advantage_stream = nn.Sequential(
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, num_actions)
                )
        else:
            if noisy:
                self.q_head = nn.Sequential(
                    NoisyLinear(512, 512, sigma_init),
                    nn.ReLU(),
                    NoisyLinear(512, num_actions, sigma_init)
                )
            else:
                self.q_head = nn.Sequential(
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, num_actions)
                )
    
    def forward(self, x):
        x = self.features(x)
        
        if self.dueling:
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            q_values = self.q_head(x)
        
        return q_values
    
    def reset_noise(self):
        if self.noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

class CategoricalDQN(nn.Module):
    """Categorical DQN for distributional RL"""
    def __init__(self, num_inputs, num_actions, atom_size, v_min, v_max, 
                 dueling=False, noisy=False, sigma_init=0.5):
        super().__init__()
        self.num_actions = num_actions
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max
        self.dueling = dueling
        self.noisy = noisy
        
        self.features = AtariCNN(num_inputs)
        self.z = torch.linspace(v_min, v_max, atom_size)
        
        if self.dueling:
            if noisy:
                self.value_stream = nn.Sequential(
                    NoisyLinear(512, 512, sigma_init),
                    nn.ReLU(),
                    NoisyLinear(512, atom_size, sigma_init)
                )
                self.advantage_stream = nn.Sequential(
                    NoisyLinear(512, 512, sigma_init),
                    nn.ReLU(),
                    NoisyLinear(512, num_actions * atom_size, sigma_init)
                )
            else:
                self.value_stream = nn.Sequential(
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, atom_size)
                )
                self.advantage_stream = nn.Sequential(
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, num_actions * atom_size)
                )
        else:
            if noisy:
                self.fc = nn.Sequential(
                    NoisyLinear(512, 512, sigma_init),
                    nn.ReLU(),
                    NoisyLinear(512, num_actions * atom_size, sigma_init)
                )
            else:
                self.fc = nn.Sequential(
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, num_actions * atom_size)
                )
    
    def forward(self, x):
        x = self.features(x)
        
        if self.dueling:
            value = self.value_stream(x).view(-1, 1, self.atom_size)
            advantage = self.advantage_stream(x).view(-1, self.num_actions, self.atom_size)
            x = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            x = self.fc(x)
            x = x.view(-1, self.num_actions, self.atom_size)
        
        x = F.softmax(x, dim=-1)
        return x
    
    def get_q_values(self, x):
        dist = self.forward(x)
        q_values = (dist * self.z.to(x.device)).sum(dim=-1)
        return q_values
    
    def reset_noise(self):
        if self.noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

class ActorCritic(nn.Module):
    """Actor-Critic network for PPO and A3C"""
    def __init__(self, num_inputs, num_actions):
        super().__init__()
        self.features = AtariCNN(num_inputs)
        
        self.actor = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        features = self.features(x)
        value = self.critic(features)
        logits = self.actor(features)
        return logits, value
    
    def get_value(self, x):
        features = self.features(x)
        return self.critic(features)
    
    def get_action(self, x, deterministic=False):
        logits, value = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            action = probs.multinomial(1).squeeze(1)
        
        log_prob = F.log_softmax(logits, dim=-1)
        log_prob = log_prob.gather(1, action.unsqueeze(1)).squeeze(1)
        
        return action, log_prob, value