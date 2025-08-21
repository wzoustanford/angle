#!/usr/bin/env python3
"""
Training script for MuZero on CartPole
Recommended way to train with customizable parameters
"""

import gymnasium
import numpy as np
from muzero_simple import SimpleMuZero
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def train_muzero_cartpole(
    num_episodes=500,
    num_simulations=24,
    batch_size=128,
    learning_rate=1e-3,
    save_every=50,
    print_every=10
):
    """
    Train MuZero on CartPole with configurable parameters
    
    Args:
        num_episodes: Total episodes to train
        num_simulations: MCTS simulations per move
        batch_size: Batch size for training
        learning_rate: Learning rate for neural networks
        save_every: Save checkpoint every N episodes
        print_every: Print stats every N episodes
    """
    
    print("="*60)
    print("MuZero Training on CartPole-v1")
    print("="*60)
    
    # Create environment
    env = gymnasium.make('CartPole-v1')
    
    # Initialize MuZero
    muzero = SimpleMuZero(
        observation_shape=env.observation_space.shape,
        action_space_size=env.action_space.n,
        num_simulations=num_simulations,
        batch_size=batch_size,
        max_moves=500,
        lr=learning_rate,
        td_steps=10,
        num_unroll_steps=5,
        discount=0.997,
        c_puct=1.25
    )
    
    print(f"Configuration:")
    print(f"  Episodes: {num_episodes}")
    print(f"  MCTS simulations: {num_simulations}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {muzero.device}")
    print("="*60)
    print()
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    training_losses = []
    best_reward = 0
    best_avg_reward = 0
    
    # Training loop
    for episode in range(num_episodes):
        # Self-play: Generate trajectory using MCTS
        trajectory = muzero.self_play_game(env)
        muzero.update_replay_buffer(trajectory)
        
        # Calculate episode metrics
        episode_reward = sum(exp.reward for exp in trajectory)
        episode_length = len(trajectory)
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Update best reward
        if episode_reward > best_reward:
            best_reward = episode_reward
            muzero.save_checkpoint('muzero_cartpole_best.pt')
        
        # Training: Update networks
        if episode >= 10:  # Start training after collecting initial data
            # More training steps as we collect more data
            num_train_steps = min(100, len(muzero.replay_buffer) // muzero.batch_size)
            
            episode_losses = []
            for _ in range(num_train_steps):
                losses = muzero.train_step()
                if losses:
                    episode_losses.append(losses['total_loss'])
            
            if episode_losses:
                avg_loss = np.mean(episode_losses)
                training_losses.append(avg_loss)
        
        # Save checkpoint periodically
        if (episode + 1) % save_every == 0:
            muzero.save_checkpoint(f'muzero_cartpole_ep{episode+1}.pt')
        
        # Print statistics
        if (episode + 1) % print_every == 0:
            recent_rewards = episode_rewards[-print_every:]
            avg_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards)
            max_recent = max(recent_rewards)
            
            # Update best average
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
            
            print(f"Episode {episode+1:4d} | "
                  f"Avg Reward: {avg_reward:6.1f} ± {std_reward:5.1f} | "
                  f"Max: {max_recent:3.0f} | "
                  f"Best Ever: {best_reward:3.0f}")
            
            if training_losses:
                recent_loss = np.mean(training_losses[-100:])
                print(f"              | Avg Loss: {recent_loss:.4f}")
    
    # Save final model
    muzero.save_checkpoint('muzero_cartpole_final.pt')
    
    # Plot training curves
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best single episode: {best_reward}")
    print(f"Best average reward: {best_avg_reward:.1f}")
    print(f"Final average (last 50): {np.mean(episode_rewards[-50:]):.1f}")
    print("\nModels saved:")
    print("  - muzero_cartpole_best.pt (best single episode)")
    print("  - muzero_cartpole_final.pt (final model)")
    print(f"  - Checkpoints every {save_every} episodes")
    
    # Create training plot
    plot_training(episode_rewards, episode_lengths, training_losses)
    
    env.close()
    return muzero, episode_rewards

def plot_training(rewards, lengths, losses):
    """Create and save training plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Episode rewards
    axes[0, 0].plot(rewards, alpha=0.6)
    axes[0, 0].plot(np.convolve(rewards, np.ones(50)/50, mode='valid'), 'r-', linewidth=2)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode lengths
    axes[0, 1].plot(lengths, alpha=0.6)
    axes[0, 1].plot(np.convolve(lengths, np.ones(50)/50, mode='valid'), 'g-', linewidth=2)
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training loss
    if losses:
        axes[1, 0].plot(losses)
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].set_xlabel('Training Steps')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Reward distribution
    axes[1, 1].hist(rewards, bins=30, edgecolor='black')
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].axvline(np.mean(rewards), color='r', linestyle='--', label=f'Mean: {np.mean(rewards):.1f}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('muzero_training_curves.png', dpi=100)
    print("\n📊 Training curves saved to muzero_training_curves.png")

def evaluate_model(checkpoint_path, num_episodes=10):
    """Evaluate a trained model"""
    
    print(f"\nEvaluating model: {checkpoint_path}")
    
    env = gymnasium.make('CartPole-v1')
    
    muzero = SimpleMuZero(
        observation_shape=env.observation_space.shape,
        action_space_size=env.action_space.n,
        num_simulations=100  # More simulations for evaluation
    )
    
    muzero.load_checkpoint(checkpoint_path)
    
    eval_rewards = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # For evaluation, don't add exploration noise
            action_probs, _ = muzero.run_mcts(obs, add_exploration_noise=False)
            action = np.argmax(action_probs)  # Greedy action
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        eval_rewards.append(total_reward)
        print(f"  Episode {episode+1}: {total_reward:.0f}")
    
    print(f"Average: {np.mean(eval_rewards):.1f} ± {np.std(eval_rewards):.1f}")
    env.close()
    
    return eval_rewards

if __name__ == "__main__":
    # Train MuZero
    muzero, rewards = train_muzero_cartpole(
        num_episodes=500,      # Train for 500 episodes
        num_simulations=24,    # 50 MCTS simulations per move
        batch_size=128,        # Batch size for training
        learning_rate=1e-3,    # Learning rate
        save_every=100,        # Save checkpoint every 100 episodes
        print_every=5         # Print stats every 20 episodes
    )
    
    # Evaluate best model
    print("\n" + "="*60)
    print("Evaluating best model...")
    print("="*60)
    evaluate_model('muzero_cartpole_best.pt', num_episodes=10)