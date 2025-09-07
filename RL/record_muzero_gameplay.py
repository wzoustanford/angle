#!/usr/bin/env python3
"""
Script to load the best MuZero Alien model and record gameplay video
"""

import gymnasium
import ale_py
import numpy as np
from muzero_simple import SimpleMuZero
import cv2
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class AlienWrapper:
    """
    Wrapper for Alien environment with preprocessing
    """
    def __init__(self, env_name='ALE/Alien-v5', frame_stack=4, frame_skip=3):
        self.env = gymnasium.make(env_name, render_mode='rgb_array')
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.frames = deque(maxlen=frame_stack)
        
        self.action_space = self.env.action_space
        self.action_space_size = self.action_space.n
        self.observation_shape = (frame_stack, 84, 84)
        
    def reset(self):
        obs, info = self.env.reset()
        self._last_info = info
        
        processed = self._preprocess_frame(obs)
        for _ in range(self.frame_stack):
            self.frames.append(processed)
        
        return self._get_stacked_frames(), info
    
    def step(self, action):
        total_reward = 0
        done = False
        
        for _ in range(self.frame_skip):  
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        
        self._last_info = info
        processed = self._preprocess_frame(obs)
        self.frames.append(processed)
        
        return self._get_stacked_frames(), total_reward, terminated, truncated, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()
    
    def _preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        return normalized
    
    def _get_stacked_frames(self):
        return np.array(self.frames, dtype=np.float32)


def record_gameplay(checkpoint_path='muzero_alien_best.pt', video_duration=10, fps=30):
    """
    Record gameplay video using the trained MuZero model
    
    Args:
        checkpoint_path: Path to the model checkpoint
        video_duration: Duration of video in seconds
        fps: Frames per second for the video
    """
    
    print(f"Loading model from: {checkpoint_path}")
    
    # Create environment
    env = AlienWrapper('ALE/Alien-v5', frame_stack=4, frame_skip=3)
    
    # Initialize MuZero
    muzero = SimpleMuZero(
        observation_shape=env.observation_shape,
        action_space_size=env.action_space_size,
        num_simulations=50  # Good balance between quality and speed
    )
    
    # Load the best model
    muzero.load_checkpoint(checkpoint_path)
    print("Model loaded successfully!")
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('muzero_alien_gameplay.mp4', fourcc, fps, (210, 160))
    
    # Reset environment
    obs, _ = env.reset()
    
    total_frames = video_duration * fps
    frame_count = 0
    total_reward = 0
    done = False
    
    print(f"Recording {video_duration} seconds of gameplay...")
    
    while frame_count < total_frames and not done:
        # Get action from MuZero (no exploration noise for demonstration)
        action_probs, _ = muzero.run_mcts(obs, add_exploration_noise=False)
        action = np.argmax(action_probs)  # Greedy action selection
        
        # Take action in environment
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Get the raw frame from environment for video
        raw_frame = env.render()
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
        
        # Write frame to video
        video_writer.write(frame_bgr)
        frame_count += 1
        
        # Show progress
        if frame_count % fps == 0:
            print(f"  {frame_count // fps} seconds recorded, Score: {total_reward:.0f}")
    
    # Release video writer
    video_writer.release()
    env.close()
    
    print("\n" + "="*60)
    print("Recording Complete!")
    print("="*60)
    print(f"Video saved as: muzero_alien_gameplay.mp4")
    print(f"Duration: {frame_count / fps:.1f} seconds")
    print(f"Final score: {total_reward:.0f}")
    print(f"Episode ended: {done}")
    
    # Convert to a more compatible format using ffmpeg
    print("\nConverting to H.264 format for better compatibility...")
    import os
    convert_cmd = (
        "ffmpeg -i muzero_alien_gameplay.mp4 "
        "-c:v libx264 -preset medium -crf 23 "
        "-pix_fmt yuv420p -movflags +faststart "
        "muzero_alien_gameplay_h264.mp4 -y"
    )
    os.system(convert_cmd)
    print("Converted video saved as: muzero_alien_gameplay_h264.mp4")
    
    return total_reward


if __name__ == "__main__":
    # Record 10 seconds of gameplay
    score = record_gameplay(
        checkpoint_path='muzero_alien_best.pt',
        video_duration=10,
        fps=30
    )