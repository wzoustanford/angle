#!/usr/bin/env python3
"""
Test dueling networks functionality for both standard DQN and R2D2
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from config.AgentConfig import AgentConfig
from model.dqn_agent import DQNAgent
from model.dqn_network import DQN
from model.r2d2_network import R2D2Network

def test_dqn_dueling():
    """Test dueling networks in standard DQN"""
    print("=== Testing Standard DQN with Dueling Networks ===")
    
    obs_shape = (12, 210, 160)  # 4 stacked RGB frames
    n_actions = 6
    
    # Test non-dueling DQN
    dqn_standard = DQN(obs_shape, n_actions, use_dueling=False)
    print(f"Standard DQN layers: {list(dqn_standard.named_modules())[-3:]}")
    
    # Test dueling DQN
    dqn_dueling = DQN(obs_shape, n_actions, use_dueling=True)
    print(f"Dueling DQN has fc_value: {hasattr(dqn_dueling, 'fc_value')}")
    print(f"Dueling DQN has fc_advantage: {hasattr(dqn_dueling, 'fc_advantage')}")
    print(f"Dueling DQN has fc2: {hasattr(dqn_dueling, 'fc2')}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 12, 210, 160)
    
    with torch.no_grad():
        output_standard = dqn_standard(dummy_input)
        output_dueling = dqn_dueling(dummy_input)
    
    print(f"Standard DQN output shape: {output_standard.shape}")
    print(f"Dueling DQN output shape: {output_dueling.shape}")
    
    assert output_standard.shape == output_dueling.shape == (2, 6), "Output shapes should match"
    print("✓ DQN dueling networks test passed\n")
    
    return True

def test_agent_dueling_integration():
    """Test dueling networks integration in DQN agent"""
    print("=== Testing DQN Agent with Dueling Networks ===")
    
    # Test standard DQN without dueling
    config_standard = AgentConfig()
    config_standard.use_dueling = False
    config_standard.device = 'cpu'
    
    agent_standard = DQNAgent(config_standard)
    print("Standard DQN agent created successfully")
    
    # Test standard DQN with dueling
    config_dueling = AgentConfig()
    config_dueling.use_dueling = True
    config_dueling.device = 'cpu'
    
    agent_dueling = DQNAgent(config_dueling)
    print("Dueling DQN agent created successfully")
    
    # Test action selection works for both
    dummy_state = np.random.rand(12, 210, 160).astype(np.float32)
    
    action_standard = agent_standard.select_action(dummy_state)
    action_dueling = agent_dueling.select_action(dummy_state)
    
    print(f"Standard DQN action: {action_standard}")
    print(f"Dueling DQN action: {action_dueling}")
    
    assert 0 <= action_standard < 6, "Action should be valid"
    assert 0 <= action_dueling < 6, "Action should be valid"
    
    print("✓ DQN Agent dueling integration test passed\n")
    
    return True

def test_r2d2_dueling():
    """Test that R2D2 already has dueling architecture"""
    print("=== Testing R2D2 Dueling Architecture ===")
    
    config = AgentConfig()
    config.use_r2d2 = True
    config.use_dueling = True  # Should work with R2D2
    config.device = 'cpu'
    
    agent = DQNAgent(config)
    print("R2D2 agent with dueling created successfully")
    
    # Check that R2D2 network has dueling components
    r2d2_net = agent.q_network
    has_value = hasattr(r2d2_net, 'fc_value')
    has_advantage = hasattr(r2d2_net, 'fc_advantage')
    
    print(f"R2D2 has fc_value: {has_value}")
    print(f"R2D2 has fc_advantage: {has_advantage}")
    
    assert has_value and has_advantage, "R2D2 should have dueling architecture"
    
    # Test action selection
    dummy_state = np.random.rand(12, 210, 160).astype(np.float32)
    action = agent.select_action(dummy_state)
    print(f"R2D2 action: {action}")
    
    assert 0 <= action < 6, "R2D2 action should be valid"
    
    print("✓ R2D2 dueling architecture test passed\n")
    
    return True

def test_dueling_math():
    """Test that dueling architecture math is correct"""
    print("=== Testing Dueling Networks Math ===")
    
    obs_shape = (12, 210, 160)
    n_actions = 6
    
    dqn_dueling = DQN(obs_shape, n_actions, use_dueling=True)
    dqn_dueling.eval()
    
    dummy_input = torch.randn(3, 12, 210, 160)
    
    with torch.no_grad():
        # Get intermediate outputs
        x = dummy_input.float() / 255.0
        x = torch.relu(dqn_dueling.conv1(x))
        x = torch.relu(dqn_dueling.conv2(x))
        x = torch.relu(dqn_dueling.conv3(x))
        x = x.view(x.size(0), -1)
        features = torch.relu(dqn_dueling.fc1(x))
        
        value = dqn_dueling.fc_value(features)  # (batch, 1)
        advantage = dqn_dueling.fc_advantage(features)  # (batch, n_actions)
        
        # Manual dueling computation
        q_manual = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Network output
        q_network = dqn_dueling(dummy_input)
    
    # Check that manual computation matches network output
    diff = torch.abs(q_manual - q_network).max().item()
    print(f"Max difference between manual and network computation: {diff}")
    
    assert diff < 1e-6, "Manual and network dueling computation should match"
    
    # Check that advantage values sum to approximately zero after mean subtraction
    advantage_centered = advantage - advantage.mean(dim=1, keepdim=True)
    advantage_sums = advantage_centered.sum(dim=1)
    max_sum = torch.abs(advantage_sums).max().item()
    print(f"Max absolute sum of centered advantages: {max_sum}")
    
    assert max_sum < 1e-5, "Centered advantages should sum to approximately zero"
    
    print("✓ Dueling networks math test passed\n")
    
    return True

def main():
    """Run all dueling networks tests"""
    print("Dueling Networks Test Suite")
    print("=" * 40)
    
    results = []
    
    try:
        results.append(test_dqn_dueling())
        results.append(test_agent_dueling_integration())
        results.append(test_r2d2_dueling())
        results.append(test_dueling_math())
    except Exception as e:
        print(f"Test failed with error: {e}")
        return 1
    
    passed = sum(results)
    total = len(results)
    
    print("=" * 40)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All dueling networks tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())