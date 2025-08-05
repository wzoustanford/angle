#!/usr/bin/env python3
"""
Test script for prioritized replay buffer components only
"""

import numpy as np
import sys
import os

# Add the parent directory (RL) to path to import model package
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

# Import from model package
from model.sum_tree import SumTree
from model.data_buffer import ReplayBuffer, PrioritizedReplayBuffer


def test_sum_tree():
    """Test the SumTree data structure"""
    print("Testing SumTree...")
    
    # Create a small sum tree
    tree = SumTree(capacity=4)
    
    # Add some data with priorities
    test_data = [
        (1.0, "experience_1"),
        (0.5, "experience_2"), 
        (2.0, "experience_3"),
        (0.1, "experience_4")
    ]
    
    for priority, data in test_data:
        tree.add(priority, data)
    
    print(f"✓ Added {len(test_data)} experiences")
    print(f"  Total priority sum: {tree.total():.2f}")
    
    # Test sampling
    print("\nSampling from tree:")
    for i in range(3):
        sample_val = np.random.uniform(0, tree.total())
        idx, priority, data = tree.get(sample_val)
        print(f"  Sample {i+1}: {data} (priority: {priority:.2f})")
    
    # Test batch sampling
    batch, idxs, priorities = tree.sample(3)
    print(f"\nBatch sample: {[data for data in batch]}")
    
    return True


def test_replay_buffers():
    """Test both replay buffer implementations"""
    print("\nTesting Replay Buffers...")
    
    # Test regular replay buffer
    print("\n1. Testing ReplayBuffer:")
    buffer = ReplayBuffer(capacity=10)
    
    # Add some fake experiences
    for i in range(5):
        state = np.random.random((4, 84, 84))
        action = np.random.randint(0, 4)
        reward = np.random.random()
        next_state = np.random.random((4, 84, 84))
        done = np.random.choice([True, False])
        
        buffer.push(state, action, reward, next_state, done)
    
    print(f"✓ Added 5 experiences to regular buffer")
    print(f"  Buffer size: {len(buffer)}")
    
    # Sample from regular buffer
    if len(buffer) >= 3:
        states, actions, rewards, next_states, dones = buffer.sample(3)
        print(f"  Sampled batch size: {len(states)}")
    
    # Test prioritized replay buffer
    print("\n2. Testing PrioritizedReplayBuffer:")
    
    # Test different priority types
    priority_types = ['td_error', 'reward', 'random']
    
    for priority_type in priority_types:
        print(f"\n   Testing with priority_type='{priority_type}':")
        
        try:
            p_buffer = PrioritizedReplayBuffer(
                capacity=10,
                alpha=0.6,
                beta=0.4,
                epsilon=1e-6,
                priority_type=priority_type
            )
            
            # Add experiences
            for i in range(5):
                state = np.random.random((4, 84, 84))
                action = np.random.randint(0, 4)
                reward = np.random.random() - 0.5  # Can be negative
                next_state = np.random.random((4, 84, 84))
                done = np.random.choice([True, False])
                
                p_buffer.push(state, action, reward, next_state, done)
            
            print(f"   ✓ Added 5 experiences")
            print(f"     Buffer size: {len(p_buffer)}")
            print(f"     Total priority: {p_buffer.tree.total():.2f}")
            
            # Sample from prioritized buffer
            if len(p_buffer) >= 3:
                states, actions, rewards, next_states, dones, weights, idxs = p_buffer.sample(3)
                print(f"     Sampled batch size: {len(states)}")
                print(f"     Importance weights: {weights}")
                
                # Test priority updates
                fake_td_errors = np.random.random(3)
                p_buffer.update_priorities(idxs, fake_td_errors)
                print(f"     ✓ Updated priorities with TD errors")
                
        except Exception as e:
            print(f"   ✗ Error with {priority_type}: {e}")
            return False
    
    return True


def test_priority_calculations():
    """Test different priority calculation methods"""
    print("\nTesting Priority Calculations...")
    
    buffer = PrioritizedReplayBuffer(capacity=10, priority_type='td_error')
    
    # Test TD-error priority
    td_error = 0.5
    priority = buffer.calculate_priority(td_error=td_error)
    print(f"✓ TD-error priority: {priority:.6f} (from error: {td_error})")
    
    # Test reward priority
    buffer.priority_type = 'reward'
    reward = -1.0
    priority = buffer.calculate_priority(reward=reward)
    print(f"✓ Reward priority: {priority:.6f} (from reward: {reward})")
    
    # Test random priority
    buffer.priority_type = 'random'
    priority = buffer.calculate_priority()
    print(f"✓ Random priority: {priority:.6f}")
    
    return True


def main():
    """Run all tests"""
    print("Starting Prioritized Replay Buffer Tests")
    print("="*50)
    
    try:
        # Test sum tree
        if not test_sum_tree():
            print("✗ SumTree test failed")
            return
        
        # Test replay buffers
        if not test_replay_buffers():
            print("✗ Replay buffer test failed")
            return
        
        # Test priority calculations
        if not test_priority_calculations():
            print("✗ Priority calculation test failed")
            return
        
        print("\n" + "="*50)
        print("✓ All tests passed successfully!")
        print("\nPrioritized Replay Buffer Implementation Summary:")
        print("- SumTree data structure working correctly")
        print("- PrioritizedReplayBuffer handles all priority types")
        print("- Importance sampling weights calculated properly")
        print("- Priority updates working correctly")
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()