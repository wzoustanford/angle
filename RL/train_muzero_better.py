#!/usr/bin/env python3
"""
Train MuZero for better performance on CartPole
"""

import gymnasium
import numpy as np
from muzero_simple import SimpleMuZero
import warnings
warnings.filterwarnings('ignore')

def train_muzero_properly():
    """Train MuZero with more episodes for better performance"""
    print("Training MuZero on CartPole-v1 for better performance...")
    
    # Create environment
    env = gymnasium.make('CartPole-v1')
    
    # Get environment specs
    observation_shape = env.observation_space.shape
    action_space_size = env.action_space.n
    
    # Initialize MuZero
    muzero = SimpleMuZero(
        observation_shape=observation_shape,
        action_space_size=action_space_size,
        num_simulations=50,
        batch_size=64,
        max_moves=500,
        lr=3e-4,
        td_steps=10,
        num_unroll_steps=5
    )
    
    print(f"Device: {muzero.device}")
    print("Training for 200 episodes...\n")
    
    rewards_history = []
    best_reward = 0
    
    for episode in range(200):
        # Self-play
        trajectory = muzero.self_play_game(env)
        muzero.update_replay_buffer(trajectory)
        
        episode_reward = sum(exp.reward for exp in trajectory)
        rewards_history.append(episode_reward)
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            muzero.save_checkpoint('muzero_cartpole_best.pt')
        
        # Training after collecting initial data
        if episode >= 10:
            # Train more intensively
            num_train_steps = min(50, len(muzero.replay_buffer) // muzero.batch_size)
            for _ in range(num_train_steps):
                losses = muzero.train_step()
        
        # Print progress
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(rewards_history[-20:])
            max_recent = max(rewards_history[-20:])
            print(f"Episode {episode+1:3d}: Avg Reward={avg_reward:6.1f}, "
                  f"Max Recent={max_recent:3.0f}, Best Ever={best_reward:3.0f}")
    
    # Save final model
    muzero.save_checkpoint('muzero_cartpole_final.pt')
    
    print(f"\n✅ Training complete!")
    print(f"Best reward achieved: {best_reward}")
    print(f"Final average (last 20): {np.mean(rewards_history[-20:]):.1f}")
    print(f"Models saved: muzero_cartpole_best.pt and muzero_cartpole_final.pt")
    
    env.close()
    return muzero

if __name__ == "__main__":
    muzero = train_muzero_properly()