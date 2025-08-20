#!/usr/bin/env python3
"""
Play trained MuZero model and save as video
"""

import gymnasium
import numpy as np
from muzero_simple import SimpleMuZero
import warnings
import imageio
import os
warnings.filterwarnings('ignore')

def play_and_record(checkpoint_path='test_muzero_checkpoint.pt', 
                    num_episodes=3, 
                    video_path='muzero_cartpole.mp4'):
    """
    Play trained MuZero model and record video
    """
    print(f"Loading model from {checkpoint_path}...")
    
    # Create environment with render mode for recording
    env = gymnasium.make('CartPole-v1', render_mode='rgb_array')
    
    # Get environment specs
    observation_shape = env.observation_space.shape
    action_space_size = env.action_space.n
    
    # Initialize MuZero
    muzero = SimpleMuZero(
        observation_shape=observation_shape,
        action_space_size=action_space_size,
        num_simulations=50,  # More simulations for better play
        device='cuda' if os.path.exists('/usr/local/cuda') else 'cpu'
    )
    
    # Load trained model if it exists
    if os.path.exists(checkpoint_path):
        muzero.load_checkpoint(checkpoint_path)
        print(f"✅ Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"⚠️ No checkpoint found at {checkpoint_path}, using untrained model")
    
    print(f"\nPlaying {num_episodes} episodes and recording to {video_path}...")
    
    # Collect frames from all episodes
    all_frames = []
    episode_rewards = []
    
    for episode in range(num_episodes):
        frames = []
        episode_reward = 0
        steps = 0
        
        # Reset environment
        observation, info = env.reset()
        
        # Capture initial frame
        frame = env.render()
        frames.append(frame)
        
        done = False
        while not done and steps < 500:  # Max 500 steps per episode
            # Use MCTS to select action
            action_probs = muzero.run_mcts(observation)
            
            # Select action with highest probability (greedy)
            action = np.argmax(action_probs)
            
            # Take action
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Capture frame
            frame = env.render()
            frames.append(frame)
            
            episode_reward += reward
            steps += 1
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward={episode_reward:.0f}, Steps={steps}")
        
        # Add episode separator (black frames)
        if episode < num_episodes - 1:
            for _ in range(10):  # 10 black frames between episodes
                frames.append(np.zeros_like(frames[0]))
        
        all_frames.extend(frames)
    
    env.close()
    
    # Save video
    print(f"\nSaving video to {video_path}...")
    
    # Create video writer with appropriate fps
    fps = 30  # 30 frames per second for smooth playback
    
    with imageio.get_writer(video_path, fps=fps) as writer:
        for frame in all_frames:
            writer.append_data(frame)
    
    print(f"✅ Video saved to {video_path}")
    print(f"\nStatistics:")
    print(f"  Average reward: {np.mean(episode_rewards):.1f}")
    print(f"  Max reward: {np.max(episode_rewards):.0f}")
    print(f"  Min reward: {np.min(episode_rewards):.0f}")
    print(f"  Total frames: {len(all_frames)}")
    print(f"  Video duration: {len(all_frames)/fps:.1f} seconds")
    
    return video_path

def create_comparison_video():
    """
    Create a comparison video showing untrained vs trained model
    """
    print("Creating comparison video...")
    
    # First, let's train a model briefly if needed
    env = gymnasium.make('CartPole-v1')
    observation_shape = env.observation_space.shape
    action_space_size = env.action_space.n
    
    # Check if we have a trained model, if not, train one quickly
    if not os.path.exists('muzero_cartpole_trained.pt'):
        print("Training a model first (this will take a minute)...")
        
        muzero = SimpleMuZero(
            observation_shape=observation_shape,
            action_space_size=action_space_size,
            num_simulations=25,
            batch_size=32,
            lr=1e-3
        )
        
        # Quick training
        for episode in range(50):
            trajectory = muzero.self_play_game(env)
            muzero.update_replay_buffer(trajectory)
            
            if episode >= 10:
                for _ in range(10):
                    muzero.train_step()
            
            if episode % 10 == 0:
                reward = sum(exp.reward for exp in trajectory)
                print(f"Training Episode {episode}: Reward={reward:.0f}")
        
        muzero.save_checkpoint('muzero_cartpole_trained.pt')
        print("✅ Training complete!")
    
    env.close()
    
    # Record trained model
    video_path = play_and_record(
        checkpoint_path='muzero_cartpole_trained.pt',
        num_episodes=3,
        video_path='muzero_cartpole_gameplay.mp4'
    )
    
    return video_path

if __name__ == "__main__":
    # Try to use existing checkpoint first
    if os.path.exists('test_muzero_checkpoint.pt'):
        video_path = play_and_record(
            checkpoint_path='test_muzero_checkpoint.pt',
            num_episodes=3,
            video_path='muzero_cartpole.mp4'
        )
    else:
        # Create and record a trained model
        video_path = create_comparison_video()
    
    print(f"\n🎬 Video ready at: {video_path}")
    print("You can play it with: ffplay muzero_cartpole.mp4")
    print("Or download it to view locally")