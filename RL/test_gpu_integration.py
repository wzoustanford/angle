#!/usr/bin/env python3
"""
Simple test script to verify GPU integration works across all algorithms
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from config.AgentConfig import AgentConfig
from model.dqn_agent import DQNAgent
from model.distributed_dqn_agent import DistributedDQNAgent
from model.device_utils import get_device_manager

def test_device_manager():
    """Test basic device manager functionality"""
    print("=== Testing Device Manager ===")
    
    # Test auto-selection
    devmgr = get_device_manager()
    print(f"Auto-selected device: {devmgr.device}")
    
    # Test tensor placement
    tensor = torch.randn(2, 3)
    tensor_on_device = devmgr.to_dev(tensor)
    print(f"Tensor device: {tensor_on_device.device}")
    
    # Test forced CPU
    cpu_devmgr = get_device_manager('cpu')
    print(f"CPU device: {cpu_devmgr.device}")
    
    print("✓ Device manager tests passed\n")

def test_dqn_gpu():
    """Test DQN agent with GPU support"""
    print("=== Testing DQN Agent GPU Support ===")
    
    config = AgentConfig()
    config.device = None  # Auto-select
    
    try:
        agent = DQNAgent(config)
        print(f"DQN Agent device: {agent.device}")
        print(f"Q-Network device: {next(agent.q_network.parameters()).device}")
        
        # Test action selection with dummy state
        dummy_state = np.random.rand(4, 3, 210, 160).astype(np.float32)
        action = agent.select_action(dummy_state)
        print(f"Selected action: {action}")
        
        print("✓ DQN Agent GPU tests passed\n")
        return True
        
    except Exception as e:
        print(f"✗ DQN Agent GPU test failed: {e}\n")
        return False

def test_r2d2_gpu():
    """Test R2D2 agent with GPU support"""
    print("=== Testing R2D2 Agent GPU Support ===")
    
    config = AgentConfig()
    config.use_r2d2 = True
    config.device = None  # Auto-select
    
    try:
        agent = DQNAgent(config)
        print(f"R2D2 Agent device: {agent.device}")
        print(f"R2D2 Network device: {next(agent.q_network.parameters()).device}")
        
        # Test action selection with dummy state
        dummy_state = np.random.rand(4, 3, 210, 160).astype(np.float32)
        action = agent.select_action(dummy_state)
        print(f"Selected action: {action}")
        
        print("✓ R2D2 Agent GPU tests passed\n")
        return True
        
    except Exception as e:
        print(f"✗ R2D2 Agent GPU test failed: {e}\n")
        return False

def test_distributed_gpu():
    """Test Distributed DQN agent with GPU support"""
    print("=== Testing Distributed DQN Agent GPU Support ===")
    
    config = AgentConfig()
    config.device = None  # Auto-select
    
    try:
        agent = DistributedDQNAgent(config, num_workers=2)
        print(f"Distributed Agent device: {agent.device}")
        print(f"Distributed Q-Network device: {next(agent.q_network.parameters()).device}")
        
        print("✓ Distributed Agent GPU tests passed\n")
        return True
        
    except Exception as e:
        print(f"✗ Distributed Agent GPU test failed: {e}\n")
        return False

def test_forced_cpu():
    """Test forcing CPU usage"""
    print("=== Testing Forced CPU Usage ===")
    
    config = AgentConfig()
    config.device = 'cpu'
    
    try:
        agent = DQNAgent(config)
        assert agent.device.type == 'cpu', f"Expected CPU, got {agent.device}"
        print(f"Forced CPU device: {agent.device}")
        print("✓ Forced CPU tests passed\n")
        return True
        
    except Exception as e:
        print(f"✗ Forced CPU test failed: {e}\n")
        return False

def main():
    """Run all GPU integration tests"""
    print("GPU Integration Test Suite")
    print("=" * 50)
    
    results = []
    
    # Test device manager
    test_device_manager()
    
    # Test individual components
    results.append(test_dqn_gpu())
    results.append(test_r2d2_gpu())
    results.append(test_distributed_gpu())
    results.append(test_forced_cpu())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All GPU integration tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())