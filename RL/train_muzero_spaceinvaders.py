#!/usr/bin/env python3
"""
Training script for MuZero on Space Invaders (Atari)
Space Invaders is a fixed shooter game with 6 actions
"""

import gymnasium
import ale_py
import numpy as np
from muzero_simple import SimpleMuZero
import matplotlib.pyplot as plt
import warnings
import cv2
from collections import deque
warnings.filterwarnings('ignore')

class SpaceInvadersWrapper:
    """
    Wrapper for Space Invaders environment with specific optimizations
    """
    def __init__(self, env_name='ALE/SpaceInvaders-v5', frame_stack=4, frame_skip=4, 
                 reward_transform='sqrt'):
        """
        Initialize Space Invaders environment with preprocessing
        
        Args:
            env_name: Space Invaders environment name
            frame_stack: Number of frames to stack
            frame_skip: Number of frames to repeat each action (default 4 for Space Invaders)
            reward_transform: Type of reward transformation
                - 'none': No transformation
                - 'clip': Clip to [-1, 1] (DQN-style)
                - 'sqrt': Square root scaling sign(x) * sqrt(|x|)
                - 'muzero': MuZero's h(x) = sign(x) * (sqrt(|x| + 1) - 1 + εx)
        """
        self.env = gymnasium.make(env_name, render_mode='rgb_array')
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.reward_transform = reward_transform
        self.epsilon = 0.001  # For MuZero transform
        self.frames = deque(maxlen=frame_stack)
        
        # Get action space info
        self.action_space = self.env.action_space
        self.action_space_size = self.action_space.n
        
        # Observation shape after preprocessing
        self.observation_shape = (frame_stack, 84, 84)
        
    def reset(self):
        """Reset environment and return stacked frames"""
        obs, info = self.env.reset()
        
        # Space Invaders starts automatically, no FIRE needed
        
        # Store info for tracking lives
        self._last_info = info
        
        # Preprocess frame
        processed = self._preprocess_frame(obs)
        
        # Initialize frame stack
        for _ in range(self.frame_stack):
            self.frames.append(processed)
        
        return self._get_stacked_frames(), info
    
    def step(self, action):
        """Take a step and return stacked frames"""
        # Repeat action for frame skipping
        total_reward = 0
        done = False
        lives_before = self._last_info.get('lives', 0) if hasattr(self, '_last_info') else 0
        
        # Use configurable frame skip
        for _ in range(self.frame_skip):  
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        
        # Check if we lost a life
        lives_after = info.get('lives', 0)
        # Space Invaders: game continues after losing a life (no special action needed)
        
        self._last_info = info
        
        # Preprocess and add to frame stack
        processed = self._preprocess_frame(obs)
        self.frames.append(processed)
        
        # Apply reward transformation
        transformed_reward = self._transform_reward(total_reward)
        
        return self._get_stacked_frames(), transformed_reward, terminated, truncated, info
    
    def render(self):
        """Render the environment"""
        return self.env.render()
    
    def close(self):
        """Close the environment"""
        self.env.close()
    
    def _preprocess_frame(self, frame):
        """
        Preprocess a single frame
        - Convert to grayscale
        - Resize to 84x84
        - Normalize to [0, 1]
        """
        # Convert RGB to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize to 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def _get_stacked_frames(self):
        """Get stacked frames as numpy array"""
        return np.array(self.frames, dtype=np.float32)
    
    def _transform_reward(self, reward):
        """
        Apply reward transformation based on configured method
        
        Args:
            reward: Raw reward from environment
            
        Returns:
            Transformed reward
        """
        if self.reward_transform == 'none':
            return reward
        elif self.reward_transform == 'clip':
            # DQN-style clipping
            return np.clip(reward, -1, 1)
        elif self.reward_transform == 'sqrt':
            # Simple square root scaling
            return np.sign(reward) * np.sqrt(np.abs(reward))
        elif self.reward_transform == 'muzero':
            # MuZero's h(x) = sign(x) * (sqrt(|x| + 1) - 1 + εx)
            # This preserves sign, compresses large values, and adds small linear term
            return np.sign(reward) * (np.sqrt(np.abs(reward) + 1) - 1 + self.epsilon * np.abs(reward))
        else:
            raise ValueError(f"Unknown reward transform: {self.reward_transform}")


def train_muzero_spaceinvaders(
    num_episodes=1200,
    num_simulations=30,
    batch_size=128,
    learning_rate=2e-4,
    save_every=100,
    print_every=10,
    max_moves_per_episode=10000,
    frame_stack=4,
    frame_skip=4,
    reward_transform='sqrt',
    dirichlet_alpha=0.25
):
    """
    Train MuZero on Space Invaders with configurable parameters
    
    Args:
        num_episodes: Total episodes to train
        num_simulations: MCTS simulations per move
        batch_size: Batch size for training
        learning_rate: Learning rate for neural networks
        save_every: Save checkpoint every N episodes
        print_every: Print stats every N episodes
        max_moves_per_episode: Maximum moves per episode
        frame_stack: Number of frames to stack
        frame_skip: Number of frames to repeat each action
        reward_transform: Reward transformation method ('none', 'clip', 'sqrt', 'muzero')
        dirichlet_alpha: Dirichlet noise parameter for exploration
    """
    
    print("="*60)
    print("MuZero Training on Space Invaders")
    print("="*60)
    
    # Create wrapped environment
    env = SpaceInvadersWrapper('ALE/SpaceInvaders-v5', frame_stack=frame_stack, 
                                frame_skip=frame_skip, reward_transform=reward_transform)
    
    # Initialize MuZero with Space Invaders-specific settings
    muzero = SimpleMuZero(
        observation_shape=env.observation_shape,  # (4, 84, 84) for stacked frames
        action_space_size=env.action_space_size,  # 6 actions for Space Invaders
        num_simulations=num_simulations,
        batch_size=batch_size,
        max_moves=max_moves_per_episode,
        lr=learning_rate,
        td_steps=10,
        num_unroll_steps=5,
        discount=0.997,
        c_puct=1.25,
        dirichlet_alpha=dirichlet_alpha,  # Space Invaders-specific alpha
        exploration_fraction=0.25  # Standard exploration fraction
    )
    
    print(f"Configuration:")
    print(f"  Environment: Space Invaders")
    print(f"  Observation shape: {env.observation_shape}")
    print(f"  Action space: {env.action_space_size} actions")
    print(f"  Frame stack: {frame_stack}")
    print(f"  Frame skip: {frame_skip}")
    print(f"  Reward transform: {reward_transform}")
    print(f"  Dirichlet alpha: {dirichlet_alpha}")
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
            muzero.save_checkpoint('muzero_spaceinvaders_best.pt')
        
        # Training: Update networks
        if episode >= 10:  # Start training after collecting initial data
            # Scale training steps based on replay buffer size
            num_train_steps = min(200, len(muzero.replay_buffer) // muzero.batch_size)
            
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
            muzero.save_checkpoint(f'muzero_spaceinvaders_ep{episode+1}.pt')
        
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
                  f"Avg Score: {avg_reward:7.1f} ± {std_reward:6.1f} | "
                  f"Max: {max_recent:5.0f} | "
                  f"Best: {best_reward:5.0f}")
            
            if training_losses:
                recent_loss = np.mean(training_losses[-100:])
                print(f"              | Avg Loss: {recent_loss:.4f} | "
                      f"Buffer Size: {len(muzero.replay_buffer)}")
    
    # Save final model
    muzero.save_checkpoint('muzero_spaceinvaders_final.pt')
    
    # Plot training curves
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best single episode: {best_reward:.0f} points")
    print(f"Best average score: {best_avg_reward:.1f}")
    print(f"Final average (last 50): {np.mean(episode_rewards[-50:]):.1f}")
    print("\nModels saved:")
    print("  - muzero_spaceinvaders_best.pt (best single episode)")
    print("  - muzero_spaceinvaders_final.pt (final model)")
    print(f"  - Checkpoints every {save_every} episodes")
    
    # Create training plot
    plot_training(episode_rewards, episode_lengths, training_losses)
    
    env.close()
    return muzero, episode_rewards

def plot_training(rewards, lengths, losses):
    """Create and save training plots for Space Invaders"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('MuZero Space Invaders Training', fontsize=16)
    
    # Episode scores
    axes[0, 0].plot(rewards, alpha=0.6, label='Episode Score')
    if len(rewards) > 50:
        axes[0, 0].plot(np.convolve(rewards, np.ones(50)/50, mode='valid'), 
                       'r-', linewidth=2, label='50-Episode Average')
    axes[0, 0].set_title('Episode Scores')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode lengths
    axes[0, 1].plot(lengths, alpha=0.6, label='Episode Length')
    if len(lengths) > 50:
        axes[0, 1].plot(np.convolve(lengths, np.ones(50)/50, mode='valid'), 
                       'g-', linewidth=2, label='50-Episode Average')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Frames')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training loss
    if losses:
        axes[1, 0].plot(losses)
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].set_xlabel('Training Steps')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Score distribution
    axes[1, 1].hist(rewards, bins=30, edgecolor='black')
    axes[1, 1].set_title('Score Distribution')
    axes[1, 1].set_xlabel('Score')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].axvline(np.mean(rewards), color='r', linestyle='--', 
                      label=f'Mean: {np.mean(rewards):.1f}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('muzero_spaceinvaders_training.png', dpi=100)
    print("\n📊 Training curves saved to muzero_spaceinvaders_training.png")

def evaluate_model(checkpoint_path, num_episodes=5, render=False):
    """Evaluate a trained model on Space Invaders"""
    
    print(f"\nEvaluating model: {checkpoint_path}")
    
    env = SpaceInvadersWrapper('ALE/SpaceInvaders-v5')
    
    muzero = SimpleMuZero(
        observation_shape=env.observation_shape,
        action_space_size=env.action_space_size,
        num_simulations=100  # More simulations for evaluation
    )
    
    muzero.load_checkpoint(checkpoint_path)
    
    eval_rewards = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 10000:
            # For evaluation, don't add exploration noise
            action_probs, _ = muzero.run_mcts(obs, add_exploration_noise=False)
            action = np.argmax(action_probs)  # Greedy action
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            if render and steps % 4 == 0:  # Render every 4th frame
                env.render()
        
        eval_rewards.append(total_reward)
        print(f"  Episode {episode+1}: Score={total_reward:.0f}, Steps={steps}")
    
    print(f"\nEvaluation Summary:")
    print(f"  Average Score: {np.mean(eval_rewards):.1f} ± {np.std(eval_rewards):.1f}")
    print(f"  Best Score: {max(eval_rewards):.0f}")
    print(f"  Worst Score: {min(eval_rewards):.0f}")
    
    env.close()
    return eval_rewards

if __name__ == "__main__":
    import os
    
    # Quick test mode or full training
    quick_test = True  # Set to False for full training
    
    if quick_test:
        print("🚀 Quick test mode - training for 50 episodes")
        muzero, rewards = train_muzero_spaceinvaders(
            num_episodes=50,        # Quick test
            num_simulations=10,     # Fewer simulations for speed
            batch_size=32,
            learning_rate=2e-4,
            save_every=25,
            print_every=5,
            max_moves_per_episode=2000,
            frame_skip=4,          # Standard frame skip for Space Invaders
            reward_transform='sqrt',  # Square root scaling works well
            dirichlet_alpha=0.25    # For 6 actions
        )
    else:
        print("👾 Full training mode - training for 1200 episodes")
        muzero, rewards = train_muzero_spaceinvaders(
            num_episodes=1200,      # Full training
            num_simulations=30,     # Moderate simulations (6 actions)
            batch_size=128,
            learning_rate=2e-4,
            save_every=100,
            print_every=10,
            max_moves_per_episode=10000,
            frame_skip=4,          # Standard frame skip for Space Invaders
            reward_transform='sqrt',  # Square root scaling works well
            dirichlet_alpha=0.25    # For 6 actions
        )
    
    # Evaluate best model if it exists
    if os.path.exists('muzero_spaceinvaders_best.pt'):
        print("\n" + "="*60)
        print("Evaluating best model...")
        print("="*60)
        evaluate_model('muzero_spaceinvaders_best.pt', num_episodes=3)