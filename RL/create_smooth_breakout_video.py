#!/usr/bin/env python3
"""
Create a SMOOTH 10-second video with proper frame capture
Captures ALL frames, not just one per decision
"""

import gymnasium
import ale_py
import numpy as np
from muzero_simple import SimpleMuZero
import imageio
import cv2
from collections import deque
import os
import warnings
warnings.filterwarnings('ignore')

class AtariWrapperSmooth:
    """
    Wrapper that returns ALL frames during frame skipping for smooth video
    """
    def __init__(self, env_name='ALE/Breakout-v5', frame_stack=4):
        self.env = gymnasium.make(env_name, render_mode='rgb_array')
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
        self.action_space = self.env.action_space
        self.action_space_size = self.action_space.n
        self.observation_shape = (frame_stack, 84, 84)
        
    def reset(self):
        obs, info = self.env.reset()
        
        # Fire to start and capture frame
        obs, _, _, _, info = self.env.step(1)
        self._last_info = info
        
        # Store initial render frame
        self.last_render_frame = self.env.render()
        
        processed = self._preprocess_frame(obs)
        for _ in range(self.frame_stack):
            self.frames.append(processed)
        
        return self._get_stacked_frames(), info
    
    def step_with_frames(self, action):
        """
        Step environment and return ALL frames for video
        Returns: obs, reward, terminated, truncated, info, video_frames
        """
        total_reward = 0
        done = False
        video_frames = []
        lives_before = self._last_info.get('lives', 0) if hasattr(self, '_last_info') else 0
        
        # Capture ALL frames during frame skip
        for _ in range(4):  # Frame skip
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Capture frame IMMEDIATELY after step
            frame = self.env.render()
            video_frames.append(frame)
            
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        
        # Check if lost life
        lives_after = info.get('lives', 0)
        if lives_before > 0 and lives_after < lives_before and lives_after > 0:
            obs, _, _, _, _ = self.env.step(1)  # FIRE
            # Capture FIRE frame too
            frame = self.env.render()
            video_frames.append(frame)
        
        self._last_info = info
        
        # Process for agent state
        processed = self._preprocess_frame(obs)
        self.frames.append(processed)
        
        return self._get_stacked_frames(), total_reward, terminated, truncated, info, video_frames
    
    def close(self):
        self.env.close()
    
    def _preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0
    
    def _get_stacked_frames(self):
        return np.array(self.frames, dtype=np.float32)


def create_smooth_video():
    """Create smooth 10-second video with ALL frames captured"""
    
    checkpoint_path = 'muzero_breakout_best.pt'
    video_path = 'muzero_breakout_10sec_smooth.mp4'
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Model not found: {checkpoint_path}")
        return None
    
    print("="*60)
    print("Creating SMOOTH Breakout Video")
    print("="*60)
    print("This version captures ALL frames for smooth playback")
    print()
    
    # Create environment
    env = AtariWrapperSmooth('ALE/Breakout-v5')
    
    # Load MuZero
    muzero = SimpleMuZero(
        observation_shape=env.observation_shape,
        action_space_size=env.action_space_size,
        num_simulations=50,
        device='cuda' if os.path.exists('/usr/local/cuda') else 'cpu'
    )
    
    muzero.load_checkpoint(checkpoint_path)
    print(f"✅ Model loaded from {checkpoint_path}")
    
    # Target: 10 seconds at 60 fps (native Atari framerate)
    # With frame_skip=4, each decision covers ~4 frames
    target_duration_seconds = 10
    fps = 60  # Atari native framerate
    target_frames = target_duration_seconds * fps  # 600 frames
    
    print(f"\n📹 Recording gameplay...")
    print(f"   Target: {target_frames} frames at {fps} fps ({target_duration_seconds} seconds)")
    print(f"   Note: Capturing ALL frames during frame skipping for smooth video")
    
    all_video_frames = []
    episode_count = 0
    total_score = 0
    
    while len(all_video_frames) < target_frames:
        # Start new episode
        obs, info = env.reset()
        episode_count += 1
        episode_score = 0
        current_lives = info.get('lives', 5)
        
        print(f"\nEpisode {episode_count} started - Lives: {current_lives}")
        
        # Add initial frame
        initial_frame = env.env.render()
        all_video_frames.append(initial_frame)
        
        done = False
        step_count = 0
        
        while not done and len(all_video_frames) < target_frames:
            # Get action from MuZero
            action_probs = muzero.run_mcts(obs)
            action = np.argmax(action_probs)
            
            # Step and get ALL frames
            obs, reward, terminated, truncated, info, video_frames = env.step_with_frames(action)
            done = terminated or truncated
            
            # Add ALL captured frames to video
            all_video_frames.extend(video_frames)
            
            if reward > 0:
                episode_score += reward
                total_score += reward
                print(f"  Brick broken! Score: {episode_score}")
            
            # Check lives
            new_lives = info.get('lives', 0)
            if new_lives < current_lives:
                print(f"  Lost a life! Lives: {new_lives}")
                current_lives = new_lives
            
            step_count += 1
            
            # Safety limit
            if step_count > 1000:
                break
        
        print(f"  Episode {episode_count} ended - Score: {episode_score}")
        
        # Add transition frames between episodes if needed
        if len(all_video_frames) < target_frames - 10:
            for _ in range(10):
                if len(all_video_frames) < target_frames:
                    # Black frame transition
                    all_video_frames.append(np.zeros_like(all_video_frames[-1]))
    
    env.close()
    
    # Trim to target
    all_video_frames = all_video_frames[:target_frames]
    
    # Save video at 60fps for smooth playback
    print(f"\n💾 Saving smooth video...")
    with imageio.get_writer(video_path, fps=fps) as writer:
        for frame in all_video_frames:
            writer.append_data(frame)
    
    # Also create a 30fps version for smaller file size
    video_path_30fps = 'muzero_breakout_10sec_smooth_30fps.mp4'
    # Take every other frame for 30fps
    frames_30fps = all_video_frames[::2]
    with imageio.get_writer(video_path_30fps, fps=30) as writer:
        for frame in frames_30fps:
            writer.append_data(frame)
    
    print("\n" + "="*60)
    print("✅ Smooth Videos Created!")
    print("="*60)
    print(f"📹 60 FPS Version: {video_path}")
    print(f"   Frames: {len(all_video_frames)}")
    print(f"   Duration: {len(all_video_frames)/60:.1f} seconds")
    print(f"   File size: {os.path.getsize(video_path)/1024:.1f} KB")
    print()
    print(f"📹 30 FPS Version: {video_path_30fps}")
    print(f"   Frames: {len(frames_30fps)}")
    print(f"   Duration: {len(frames_30fps)/30:.1f} seconds")
    print(f"   File size: {os.path.getsize(video_path_30fps)/1024:.1f} KB")
    print()
    print(f"📊 Gameplay Stats:")
    print(f"   Episodes: {episode_count}")
    print(f"   Total Score: {total_score}")
    print("="*60)
    
    return video_path


if __name__ == "__main__":
    video_path = create_smooth_video()