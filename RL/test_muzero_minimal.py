#!/usr/bin/env python3
"""
Minimal test to verify MuZero works
"""

import torch
import numpy as np
from config.MuZeroConfig import MuZeroConfig
from model.muzero_network import MuZeroNetwork
from model.muzero_mcts import MCTS


def test_network():
    """Test MuZero network components"""
    print("Testing MuZero network...")
    
    # Create config
    config = MuZeroConfig()
    config.action_space_size = 6  # SpaceInvaders action space
    config.observation_shape = (3, 96, 96)
    config.hidden_size = 64  # Smaller for testing
    config.support_size = 10  # Smaller for testing
    
    # Create network
    network = MuZeroNetwork(config)
    
    # Test initial inference
    obs = torch.randn(1, 3, 96, 96)
    output = network.initial_inference(obs)
    
    print(f"✓ Initial inference works")
    print(f"  State shape: {output['state'].shape}")
    print(f"  Policy logits shape: {output['policy_logits'].shape}")
    print(f"  Value: {output['value'].item():.3f}")
    
    # Test recurrent inference
    action = torch.tensor([2])
    rec_output = network.recurrent_inference(output['state'], action)
    
    print(f"✓ Recurrent inference works")
    print(f"  Reward: {rec_output['reward'].item():.3f}")
    print(f"  Next value: {rec_output['value'].item():.3f}")
    
    # Test forward with unroll
    actions = [torch.tensor([i]) for i in [0, 1, 2]]
    forward_output = network(obs, actions)
    
    print(f"✓ Forward pass with unroll works")
    print(f"  Number of steps: {len(forward_output['value'])}")
    
    return True


def test_mcts():
    """Test MCTS component"""
    print("\nTesting MCTS...")
    
    # Create config
    config = MuZeroConfig()
    config.action_space_size = 6
    config.num_simulations = 10  # Few simulations for quick test
    config.observation_shape = (3, 96, 96)
    config.hidden_size = 64
    config.support_size = 10
    
    # Create network and MCTS
    network = MuZeroNetwork(config)
    mcts = MCTS(config)
    
    # Run MCTS
    obs = torch.randn(3, 96, 96)
    result = mcts.run(obs, network, temperature=1.0, add_exploration_noise=False)
    
    print(f"✓ MCTS completed")
    print(f"  Selected action: {result['action']}")
    print(f"  Value estimate: {result['value']:.3f}")
    print(f"  Total visits: {result['visit_counts'].sum()}")
    
    return True


def main():
    print("="*60)
    print("MuZero Minimal Test")
    print("="*60)
    
    try:
        # Test network
        if test_network():
            print("\n✅ Network test passed!")
        
        # Test MCTS
        if test_mcts():
            print("\n✅ MCTS test passed!")
        
        print("\n" + "="*60)
        print("✅ MuZero implementation is working!")
        print("\nYou can train MuZero with:")
        print("  python train_muzero.py --game SpaceInvaders --iterations 100")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()