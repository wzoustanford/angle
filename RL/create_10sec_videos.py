#!/usr/bin/env python3
"""
Create 10-second videos: one with trained MuZero, one with random actions
"""

import gymnasium
import numpy as np
from muzero_simple import SimpleMuZero
import imageio
import warnings
warnings.filterwarnings('ignore')

def create_10sec_video_trained():
    """Create 10-second video with best trained MuZero model"""
    
    print("Creating 10-second video with trained MuZero (best reward: 65)...")
    
    # Create environment for recording
    env = gymnasium.make('CartPole-v1', render_mode='rgb_array')
    
    # Initialize and load the best model
    muzero = SimpleMuZero(
        observation_shape=env.observation_space.shape,
        action_space_size=env.action_space.n,
        num_simulations=20,
        device='cuda'
    )
    
    # Load the best checkpoint
    muzero.load_checkpoint('muzero_quick_best.pt')
    print("✅ Loaded best model (reward 65)")
    
    # Target: 10 seconds at 30 fps = 300 frames
    target_frames = 300
    fps = 30
    
    all_frames = []
    episode_count = 0
    total_steps = 0
    episode_rewards = []
    
    while len(all_frames) < target_frames:
        episode_count += 1
        obs, _ = env.reset()
        
        # Capture initial frame
        frame = env.render()
        all_frames.append(frame)
        
        done = False
        episode_reward = 0
        episode_steps = 0
        
        while not done and len(all_frames) < target_frames:
            # Use MCTS to select action
            action_probs = muzero.run_mcts(obs)
            # Greedy action selection for best performance
            action = np.argmax(action_probs)
            
            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Capture frame
            frame = env.render()
            all_frames.append(frame)
            
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
        
        episode_rewards.append(episode_reward)
        print(f"  Episode {episode_count}: {episode_reward:.0f} reward, {episode_steps} steps")
        
        # If episode ended and we need more frames, add a transition
        if not done and len(all_frames) < target_frames - 5:
            # Add a few black frames as transition
            for _ in range(5):
                if len(all_frames) < target_frames:
                    all_frames.append(np.zeros_like(frame))
    
    env.close()
    
    # Trim to exactly 300 frames if we have more
    all_frames = all_frames[:target_frames]
    
    # Save video
    video_path = 'muzero_trained_10sec.mp4'
    with imageio.get_writer(video_path, fps=fps) as writer:
        for frame in all_frames:
            writer.append_data(frame)
    
    duration = len(all_frames) / fps
    avg_reward = np.mean(episode_rewards) if episode_rewards else 0
    
    print(f"\n✅ Trained MuZero video saved: {video_path}")
    print(f"  Duration: {duration:.1f} seconds")
    print(f"  Episodes: {episode_count}")
    print(f"  Total steps: {total_steps}")
    print(f"  Average reward per episode: {avg_reward:.1f}")
    print(f"  Frames: {len(all_frames)}")
    
    return video_path

def create_10sec_video_random():
    """Create 10-second video with random actions"""
    
    print("\nCreating 10-second video with random actions...")
    
    # Create environment for recording
    env = gymnasium.make('CartPole-v1', render_mode='rgb_array')
    
    # Target: 10 seconds at 30 fps = 300 frames
    target_frames = 300
    fps = 30
    
    all_frames = []
    episode_count = 0
    total_steps = 0
    episode_rewards = []
    
    while len(all_frames) < target_frames:
        episode_count += 1
        obs, _ = env.reset()
        
        # Capture initial frame
        frame = env.render()
        all_frames.append(frame)
        
        done = False
        episode_reward = 0
        episode_steps = 0
        
        while not done and len(all_frames) < target_frames:
            # Random action selection
            action = env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Capture frame
            frame = env.render()
            all_frames.append(frame)
            
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
        
        episode_rewards.append(episode_reward)
        print(f"  Episode {episode_count}: {episode_reward:.0f} reward, {episode_steps} steps")
        
        # If episode ended and we need more frames, add a transition
        if not done and len(all_frames) < target_frames - 5:
            # Add a few black frames as transition
            for _ in range(5):
                if len(all_frames) < target_frames:
                    all_frames.append(np.zeros_like(frame))
    
    env.close()
    
    # Trim to exactly 300 frames if we have more
    all_frames = all_frames[:target_frames]
    
    # Save video
    video_path = 'random_agent_10sec.mp4'
    with imageio.get_writer(video_path, fps=fps) as writer:
        for frame in all_frames:
            writer.append_data(frame)
    
    duration = len(all_frames) / fps
    avg_reward = np.mean(episode_rewards) if episode_rewards else 0
    
    print(f"\n✅ Random agent video saved: {video_path}")
    print(f"  Duration: {duration:.1f} seconds")
    print(f"  Episodes: {episode_count}")
    print(f"  Total steps: {total_steps}")
    print(f"  Average reward per episode: {avg_reward:.1f}")
    print(f"  Frames: {len(all_frames)}")
    
    return video_path

def main():
    """Create both 10-second videos"""
    
    print("="*60)
    print("Creating 10-second comparison videos")
    print("="*60)
    
    # Create trained agent video
    trained_video = create_10sec_video_trained()
    
    # Create random agent video
    random_video = create_10sec_video_random()
    
    print("\n" + "="*60)
    print("✅ Both videos created successfully!")
    print("="*60)
    print(f"1. Trained MuZero: {trained_video}")
    print(f"2. Random Agent:   {random_video}")
    print("\nYou can play them with:")
    print(f"  ffplay {trained_video}")
    print(f"  ffplay {random_video}")

if __name__ == "__main__":
    main()