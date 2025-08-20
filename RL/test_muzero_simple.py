#!/usr/bin/env python3
"""
Test script for SimpleMuZero implementation
"""

import gymnasium
import numpy as np
from muzero_simple import SimpleMuZero
import warnings
warnings.filterwarnings('ignore')

def test_muzero():
    """Quick test of MuZero on CartPole"""
    print("Testing SimpleMuZero on CartPole-v1...")
    
    # Create environment
    env = gymnasium.make('CartPole-v1')
    
    # Get environment specs
    observation_shape = env.observation_space.shape
    action_space_size = env.action_space.n
    
    print(f"Environment: CartPole-v1")
    print(f"Observation shape: {observation_shape}")
    print(f"Action space size: {action_space_size}")
    
    # Initialize MuZero with smaller settings for quick test
    muzero = SimpleMuZero(
        observation_shape=observation_shape,
        action_space_size=action_space_size,
        num_simulations=10,  # Fewer simulations for speed
        batch_size=16,
        max_moves=200,
        lr=1e-3
    )
    
    print(f"Device: {muzero.device}")
    print("\nRunning self-play and training...")
    
    # Collect initial trajectories
    rewards = []
    for episode in range(20):
        # Self-play
        trajectory = muzero.self_play_game(env)
        muzero.update_replay_buffer(trajectory)
        
        episode_reward = sum(exp.reward for exp in trajectory)
        rewards.append(episode_reward)
        
        # Start training after collecting some data
        if episode >= 5:
            losses = None
            for _ in range(5):  # Few training steps per episode
                losses = muzero.train_step()
                if losses:
                    break
            
            if episode % 5 == 0:
                avg_reward = np.mean(rewards[-5:])
                print(f"Episode {episode}: Avg Reward={avg_reward:.1f}, Steps={len(trajectory)}")
                if losses:
                    print(f"  Loss: {losses['total_loss']:.4f} (P:{losses['policy_loss']:.4f}, V:{losses['value_loss']:.4f}, R:{losses['reward_loss']:.4f})")
    
    print("\n✅ Test completed successfully!")
    
    # Test MCTS planning
    print("\nTesting MCTS planning...")
    obs, _ = env.reset()
    action_probs = muzero.run_mcts(obs)
    print(f"Action probabilities from MCTS: {action_probs}")
    
    # Test save/load
    print("\nTesting checkpoint save/load...")
    muzero.save_checkpoint('test_muzero_checkpoint.pt')
    print("Checkpoint saved successfully!")
    
    # Create new instance and load
    muzero2 = SimpleMuZero(
        observation_shape=observation_shape,
        action_space_size=action_space_size,
        num_simulations=10
    )
    muzero2.load_checkpoint('test_muzero_checkpoint.pt')
    print("Checkpoint loaded successfully!")
    
    print("\n🎉 All tests passed!")

if __name__ == "__main__":
    test_muzero()