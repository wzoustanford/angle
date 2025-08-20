#!/usr/bin/env python3
"""
Generate 10-second video using muzero_cartpole_best.pt
"""

import gymnasium
import numpy as np
from muzero_simple import SimpleMuZero
import imageio
import os
import warnings
warnings.filterwarnings('ignore')

def create_10sec_video_best_model():
    """Create 10-second video with muzero_cartpole_best.pt model"""
    
    checkpoint_path = 'muzero_cartpole_best.pt'
    
    # Check if model exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Model not found: {checkpoint_path}")
        print("Please train the model first using: python train_muzero_cartpole.py")
        return None
    
    print(f"Loading model from {checkpoint_path}...")
    
    # Create environment for recording
    env = gymnasium.make('CartPole-v1', render_mode='rgb_array')
    
    # Initialize MuZero and load the best model
    muzero = SimpleMuZero(
        observation_shape=env.observation_space.shape,
        action_space_size=env.action_space.n,
        num_simulations=50,  # Good balance of quality and speed
        device='cuda' if os.path.exists('/usr/local/cuda') else 'cpu'
    )
    
    # Load the best checkpoint
    muzero.load_checkpoint(checkpoint_path)
    print("✅ Loaded best model successfully")
    
    # Target: 10 seconds at 30 fps = 300 frames
    target_frames = 300
    fps = 30
    
    print(f"\n📹 Recording 10-second gameplay video...")
    print(f"Target: {target_frames} frames at {fps} fps")
    
    all_frames = []
    episode_count = 0
    total_reward = 0
    episode_rewards = []
    episode_lengths = []
    
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
            
            # You could also use sampling for more variety:
            # action = np.random.choice(len(action_probs), p=action_probs)
            
            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Capture frame
            frame = env.render()
            all_frames.append(frame)
            
            episode_reward += reward
            episode_steps += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        total_reward += episode_reward
        
        print(f"  Episode {episode_count}: {episode_reward:3.0f} reward, {episode_steps:3} steps")
        
        # Add transition frames between episodes if needed
        if not done and len(all_frames) < target_frames - 10:
            # Add black frames as transition
            for _ in range(5):
                if len(all_frames) < target_frames:
                    all_frames.append(np.zeros_like(frame))
    
    env.close()
    
    # Trim to exactly target frames
    all_frames = all_frames[:target_frames]
    
    # Save video
    video_path = 'muzero_best_model_10sec.mp4'
    
    print(f"\n💾 Saving video to {video_path}...")
    
    with imageio.get_writer(video_path, fps=fps) as writer:
        for frame in all_frames:
            writer.append_data(frame)
    
    # Calculate statistics
    duration = len(all_frames) / fps
    avg_reward = np.mean(episode_rewards) if episode_rewards else 0
    avg_length = np.mean(episode_lengths) if episode_lengths else 0
    max_reward = max(episode_rewards) if episode_rewards else 0
    
    print("\n" + "="*60)
    print("✅ Video created successfully!")
    print("="*60)
    print(f"📹 Video: {video_path}")
    print(f"⏱️  Duration: {duration:.1f} seconds ({len(all_frames)} frames)")
    print(f"🎮 Episodes played: {episode_count}")
    print(f"📊 Performance:")
    print(f"   - Average reward: {avg_reward:.1f}")
    print(f"   - Average episode length: {avg_length:.1f} steps")
    print(f"   - Best episode: {max_reward:.0f} reward")
    print(f"   - Total reward: {total_reward:.0f}")
    print("\n🎬 Play the video with: ffplay " + video_path)
    
    return video_path

if __name__ == "__main__":
    video_path = create_10sec_video_best_model()
    
    if video_path is None:
        print("\n⚠️  No video created. Training the model first...")
        print("\nRunning quick training (this will take about 2 minutes)...")
        
        # Quick training if model doesn't exist
        from train_muzero_cartpole import train_muzero_cartpole
        
        muzero, rewards = train_muzero_cartpole(
            num_episodes=100,      # Quick training
            num_simulations=20,    # Fewer simulations for speed
            batch_size=64,
            learning_rate=1e-3,
            save_every=50,
            print_every=10
        )
        
        print("\n✅ Training complete! Now creating video...")
        video_path = create_10sec_video_best_model()