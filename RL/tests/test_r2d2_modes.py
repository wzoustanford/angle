#!/usr/bin/env python3
"""
Test script to demonstrate how to run different agent modes from train.py
"""

import sys
import os

# Add the parent directory (RL) to path to import packages
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

from config.AgentConfig import AgentConfig
from model import DQNAgent


def test_standard_dqn():
    """Test standard DQN mode"""
    print("Testing Standard DQN Mode")
    print("-" * 30)
    
    config = AgentConfig()
    config.use_r2d2 = False
    config.use_prioritized_replay = False
    config.memory_size = 1000
    config.min_replay_size = 100
    
    try:
        agent = DQNAgent(config)
        print("✓ Standard DQN agent created successfully")
        print(f"  Network type: {type(agent.q_network).__name__}")
        print(f"  Buffer type: {type(agent.replay_buffer).__name__}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_prioritized_dqn():
    """Test DQN with prioritized replay"""
    print("\nTesting DQN + Prioritized Replay")
    print("-" * 35)
    
    config = AgentConfig()
    config.use_r2d2 = False
    config.use_prioritized_replay = True
    config.priority_type = 'td_error'
    config.memory_size = 1000
    config.min_replay_size = 100
    
    try:
        agent = DQNAgent(config)
        print("✓ Prioritized DQN agent created successfully")
        print(f"  Network type: {type(agent.q_network).__name__}")
        print(f"  Buffer type: {type(agent.replay_buffer).__name__}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_r2d2_uniform():
    """Test R2D2 with uniform replay"""
    print("\nTesting R2D2 + Uniform Replay")
    print("-" * 30)
    
    config = AgentConfig()
    config.use_r2d2 = True
    config.use_prioritized_replay = False
    config.sequence_length = 80
    config.burn_in_length = 40
    config.lstm_size = 512
    config.memory_size = 1000
    config.min_replay_size = 100
    
    try:
        agent = DQNAgent(config)
        print("✓ R2D2 agent created successfully")
        print(f"  Network type: {type(agent.q_network).__name__}")
        print(f"  Buffer type: {type(agent.replay_buffer).__name__}")
        print(f"  LSTM size: {config.lstm_size}")
        print(f"  Sequence length: {config.sequence_length}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_r2d2_prioritized():
    """Test R2D2 with prioritized replay (full R2D2)"""
    print("\nTesting R2D2 + Prioritized Replay (Full R2D2)")
    print("-" * 45)
    
    config = AgentConfig()
    config.use_r2d2 = True
    config.use_prioritized_replay = True
    config.priority_type = 'td_error'
    config.sequence_length = 80
    config.burn_in_length = 40
    config.lstm_size = 512
    config.memory_size = 1000
    config.min_replay_size = 100
    
    try:
        agent = DQNAgent(config)
        print("✓ Full R2D2 agent created successfully")
        print(f"  Network type: {type(agent.q_network).__name__}")
        print(f"  Buffer type: {type(agent.replay_buffer).__name__}")
        print(f"  LSTM size: {config.lstm_size}")
        print(f"  Sequence length: {config.sequence_length}")
        print(f"  Priority type: {config.priority_type}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def show_train_py_examples():
    """Show how to configure train.py for different modes"""
    print("\n" + "=" * 60)
    print("HOW TO USE WITH train.py")
    print("=" * 60)
    
    print("""
To run different modes, modify config/AgentConfig.py:

1. Standard DQN (original):
   use_r2d2 = False
   use_prioritized_replay = False

2. DQN + Prioritized Replay:
   use_r2d2 = False  
   use_prioritized_replay = True

3. R2D2 + Uniform Replay:
   use_r2d2 = True
   use_prioritized_replay = False

4. Full R2D2 (recommended):
   use_r2d2 = True
   use_prioritized_replay = True
   priority_type = 'td_error'
   sequence_length = 80
   burn_in_length = 40
   lstm_size = 512

Then run: python train.py
""")


def main():
    """Run all tests"""
    print("R2D2 Integration Test")
    print("=" * 60)
    
    tests = [
        test_standard_dqn,
        test_prioritized_dqn,
        test_r2d2_uniform,
        test_r2d2_prioritized
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("✓ All agent modes working correctly!")
        show_train_py_examples()
    else:
        print("✗ Some tests failed")


if __name__ == "__main__":
    main()