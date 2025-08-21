#!/usr/bin/env python3
"""
Self-play function for MuZero on Breakout with proper FIRE handling
"""

import numpy as np
import torch
from muzero_simple import SimpleMuZero, Experience
from train_muzero_breakout import AtariWrapper
import warnings
warnings.filterwarnings('ignore')

def self_play_game_breakout(muzero, env):
    """
    Self-play a game of Breakout using MCTS for action selection.
    Properly handles FIRE action through AtariWrapper.
    
    Args:
        muzero: SimpleMuZero instance
        env: AtariWrapper environment (handles FIRE automatically)
    
    Returns:
        Trajectory of experiences for training
    """
    trajectory = []
    
    # Reset environment - AtariWrapper automatically fires to start
    observation, info = env.reset()
    
    print(f"Starting self-play game. Initial lives: {info.get('lives', 'N/A')}")
    
    for step in range(muzero.max_moves):
        # Run MCTS to get action probabilities
        search_policy = muzero.run_mcts(observation)
        
        # Sample action (with temperature for exploration during training)
        # During training, use temperature=1 for first 30 moves, then 0
        if step < 30:
            # Exploration phase - sample from distribution
            action = np.random.choice(muzero.action_space_size, p=search_policy)
        else:
            # Exploitation phase - choose best action
            action = np.argmax(search_policy)
        
        # Execute action in environment
        # AtariWrapper handles FIRE automatically when losing a life
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store experience (value targets will be computed later)
        trajectory.append(Experience(
            observation=observation,
            action=action,
            reward=reward,
            search_policy=search_policy,
            value_target=0.0  # Will be filled during training
        ))
        
        # Log progress occasionally
        if step % 100 == 0 and step > 0:
            total_reward = sum(exp.reward for exp in trajectory)
            print(f"  Step {step}: Score={total_reward}, Lives={info.get('lives', 'N/A')}")
        
        observation = next_observation
        
        if done:
            total_reward = sum(exp.reward for exp in trajectory)
            print(f"Game ended. Final score: {total_reward}, Steps: {len(trajectory)}")
            break
    
    return trajectory


def test_self_play():
    """Test self-play with a trained or untrained model"""
    import os
    
    print("Testing MuZero self-play on Breakout...")
    print("="*60)
    
    # Create environment with automatic FIRE handling
    env = AtariWrapper('ALE/Breakout-v5', frame_stack=4)
    
    # Initialize MuZero
    muzero = SimpleMuZero(
        observation_shape=env.observation_shape,
        action_space_size=env.action_space_size,
        num_simulations=10,  # Few simulations for quick test
        max_moves=1000
    )
    
    # Load model if available
    if os.path.exists('muzero_breakout_best.pt'):
        muzero.load_checkpoint('muzero_breakout_best.pt')
        print("✅ Loaded trained model")
    else:
        print("⚠️ Using untrained model")
    
    # Run self-play
    print("\nRunning self-play game...")
    trajectory = self_play_game_breakout(muzero, env)
    
    # Analyze trajectory
    print("\n" + "="*60)
    print("Self-play Analysis:")
    print(f"  Total steps: {len(trajectory)}")
    print(f"  Total score: {sum(exp.reward for exp in trajectory):.0f}")
    
    # Action distribution
    actions = [exp.action for exp in trajectory]
    action_counts = {i: actions.count(i) for i in range(env.action_space_size)}
    print(f"  Action distribution:")
    action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
    for action, count in action_counts.items():
        percentage = (count / len(actions)) * 100 if actions else 0
        print(f"    {action_names[action]}: {count} ({percentage:.1f}%)")
    
    # Reward distribution
    rewards = [exp.reward for exp in trajectory]
    non_zero_rewards = [r for r in rewards if r > 0]
    if non_zero_rewards:
        print(f"  Bricks broken: {len(non_zero_rewards)}")
        print(f"  Points per brick: {non_zero_rewards[0] if non_zero_rewards else 0}")
    
    env.close()
    print("\n✅ Self-play test complete!")
    
    return trajectory


if __name__ == "__main__":
    # Test the self-play function
    trajectory = test_self_play()
    
    # Optional: Save trajectory for analysis
    import pickle
    with open('breakout_trajectory.pkl', 'wb') as f:
        pickle.dump(trajectory, f)
    print("\nTrajectory saved to breakout_trajectory.pkl")