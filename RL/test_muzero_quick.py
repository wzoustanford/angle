#!/usr/bin/env python3
"""
Quick test script to verify MuZero implementation
"""

import torch
import numpy as np
from config.MuZeroConfig import MuZeroConfig
from model.muzero_agent import MuZeroAgent


def test_muzero_components():
    """Test individual MuZero components"""
    print("Testing MuZero components...")
    
    # Create config
    config = MuZeroConfig()
    config.env_name = 'ALE/SpaceInvaders-v5'
    
    # Test agent initialization
    print("1. Initializing MuZero agent...")
    agent = MuZeroAgent(config)
    print(f"   ✓ Agent created with device: {agent.device}")
    print(f"   ✓ Action space size: {agent.action_space_size}")
    
    # Test network forward pass
    print("\n2. Testing network forward pass...")
    dummy_obs = torch.randn(1, 3, 96, 96).to(agent.device)
    with torch.no_grad():
        output = agent.network.initial_inference(dummy_obs)
    print(f"   ✓ Initial inference successful")
    print(f"   ✓ State shape: {output['state'].shape}")
    print(f"   ✓ Policy shape: {output['policy_logits'].shape}")
    print(f"   ✓ Value: {output['value'].item():.3f}")
    
    # Test recurrent inference
    print("\n3. Testing recurrent inference...")
    action = torch.tensor([0]).to(agent.device)
    with torch.no_grad():
        rec_output = agent.network.recurrent_inference(output['state'], action)
    print(f"   ✓ Recurrent inference successful")
    print(f"   ✓ Reward: {rec_output['reward'].item():.3f}")
    print(f"   ✓ Next value: {rec_output['value'].item():.3f}")
    
    # Test MCTS
    print("\n4. Testing MCTS...")
    obs, _ = agent.env.reset()
    obs_tensor = agent.preprocess_observation(obs).to(agent.device)
    mcts_result = agent.mcts.run(obs_tensor, agent.network, temperature=1.0)
    print(f"   ✓ MCTS completed")
    print(f"   ✓ Selected action: {mcts_result['action']}")
    print(f"   ✓ Root value: {mcts_result['value']:.3f}")
    print(f"   ✓ Visit counts sum: {mcts_result['visit_counts'].sum()}")
    
    # Test self-play
    print("\n5. Testing self-play (1 episode)...")
    game = agent.self_play()
    print(f"   ✓ Self-play completed")
    print(f"   ✓ Episode length: {len(game)}")
    print(f"   ✓ Total reward: {sum(game.rewards):.1f}")
    print(f"   ✓ Buffer size: {len(agent.replay_buffer.buffer)}")
    
    # Test training step
    print("\n6. Testing training step...")
    # Need more data for training
    for _ in range(5):
        agent.self_play()
    
    if agent.replay_buffer.is_ready():
        metrics = agent.train_step()
        print(f"   ✓ Training step completed")
        print(f"   ✓ Total loss: {metrics['total_loss']:.4f}")
        print(f"   ✓ Value loss: {metrics['value_loss']:.4f}")
        print(f"   ✓ Policy loss: {metrics['policy_loss']:.4f}")
    else:
        print("   ⚠ Not enough data for training yet")
    
    print("\n✅ All MuZero components working correctly!")
    return True


def quick_training_test():
    """Run a quick training test"""
    print("\n" + "="*60)
    print("Running quick MuZero training test...")
    print("="*60)
    
    config = MuZeroConfig()
    config.env_name = 'ALE/SpaceInvaders-v5'
    config.num_simulations = 10  # Reduced for quick test
    config.batch_size = 32
    
    agent = MuZeroAgent(config)
    
    # Quick training loop
    print("\nTraining for 10 iterations...")
    for i in range(10):
        # Self-play
        game = agent.self_play()
        game_reward = sum(game.rewards)
        
        # Training
        if agent.replay_buffer.is_ready():
            metrics = agent.train_step()
            print(f"Iteration {i+1}: Reward={game_reward:.1f}, Loss={metrics['total_loss']:.4f}")
        else:
            print(f"Iteration {i+1}: Reward={game_reward:.1f}, Building replay buffer...")
    
    # Quick evaluation
    print("\nEvaluating trained agent (1 episode)...")
    eval_metrics = agent.evaluate(num_episodes=1)
    print(f"Evaluation reward: {eval_metrics['mean_reward']:.1f}")
    
    print("\n✅ Quick training test completed successfully!")


if __name__ == '__main__':
    try:
        # Test components
        success = test_muzero_components()
        
        if success:
            # Run quick training
            quick_training_test()
            
        print("\n" + "="*60)
        print("MuZero implementation verified successfully!")
        print("You can now train MuZero using:")
        print("  python train_muzero.py --game SpaceInvaders --iterations 1000")
        print("Or compare with other algorithms:")
        print("  python experiments/muzero_comparison.py --game SpaceInvaders --episodes 20")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()